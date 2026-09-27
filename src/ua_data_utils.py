#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ua_data_utils.py — UA 因果 ML 管线的单一事实源（Single Source of Truth）
=========================================================================


原管线里每个脚本各自处理 dtype、各自编码性别、各自划分交叉验证，
这是「性别字符串直接喂给线性回归」「标准化在 CV 之外做」「肾功能分层
把超滤过者判成缺失」等一系列缺陷的共同根因。本模块把这类逻辑集中到一处。

四类职责
--------
1. 编码与分层      encode_sex / assign_renal_group
   性别编码只定义一次，遇未映射取值**直接抛错**，绝不静默变成 NaN；
   肾功能分层上界放到 1000，避免 eGFR>150 的超滤过者被判成缺失。

2. 数据加载        load_merged_data
   列名清洗 → 性别编码 → 数值化 → 血液单位统一（默认 μmol/L）
   → (subject_id, time_point) 唯一性校验 → 返回 (df, meta)。
   meta 里带 synthetic_flag，用于在报告里强制打「合成数据」水印。

3. 无泄漏交叉验证   oof_predict
   标准化器放进 Pipeline，**每折内部 fit**。
   「先对全量数据 fit_transform 再送进 CV」会把测试折的信息漏进训练折，
   使折外误差被系统性低估。

4. 不确定度        cluster_bootstrap_ci / paired_delta_ci
   同一受试者的多次测量不独立，必须按**受试者整簇**有放回重采样；
   按测量条重采样会人为压低置信区间宽度。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # 仅用于类型标注；运行时不导入，见下方 _sklearn()
    from sklearn.model_selection import GroupKFold


def _sklearn():
    """
    延迟导入 scikit-learn。

    本模块的「编码 / 分层 / 加载」三块功能不需要 sklearn，
    而 ``ua_causal_specification.py``（规格脚本）只用到 ``write_csv``——
    如果这里在模块顶层 import sklearn，规格脚本就被无谓地绑上了重依赖。
    因此把 sklearn 收进函数内部按需导入。
    """
    from sklearn.base import clone
    from sklearn.model_selection import GroupKFold, KFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return clone, GroupKFold, KFold, Pipeline, StandardScaler

# --------------------------------------------------------------------------
# 常量
# --------------------------------------------------------------------------

#: 尿酸分子量 168.11 g/mol → 1 mg/dL = 59.48 μmol/L
BLOOD_UA_UMOL_PER_MGDL = 59.48

#: 性别编码表。**只在这里定义一次**，其它脚本一律调用 encode_sex。
SEX_MAP = {
    "M": 1, "m": 1, "male": 1, "Male": 1, "MALE": 1, "1": 1, 1: 1,
    "F": 0, "f": 0, "female": 0, "Female": 0, "FEMALE": 0, "0": 0, 0: 0,
}

#: 肾功能分层边界（right=False，即左闭右开）。
#: 上界取 1000 而非 150——eGFR 超过 150 的超滤过者在真实数据里确实存在，
#: 原代码上界 150 会把这些人整行判为 NaN 并静默丢出分析。
RENAL_BINS = [0.0, 60.0, 90.0, 1000.0]
RENAL_LABELS = ["eGFR<60", "eGFR 60-89", "eGFR>=90"]
RENAL_GROUP_ORDER = ["eGFR<60", "eGFR 60-89", "eGFR>=90"]

#: 验证方案排除 CKD stage 3+（eGFR<60）的入组标准，故分层分析只跑两层。
STUDY_RENAL_LABELS = ["eGFR 60-89", "eGFR>=90"]
EXCLUDED_RENAL_LABEL = "eGFR<60(方案排除)"

#: 数据表中非数值的列（其余列一律强制数值化）。
ID_COLS = {"subject_id", "time_point", "stage", "notes"}
BOOL_COLS = {"sweat_UA_censored"}

#: 管线期望出现的列，用于 meta["missing_columns"]。
EXPECTED_COLUMNS = [
    "subject_id", "time_point", "sweat_UA", "sweat_rate", "sweat_pH",
    "BMI", "sex", "age", "eGFR", "creatinine", "BUN",
    "fat_mass", "muscle_mass", "BMR", "blood_UA",
]

METRIC_NAMES = ("mae", "r2", "rmse")


# --------------------------------------------------------------------------
# 0. 通用小工具
# --------------------------------------------------------------------------

def setup_console() -> None:
    """
    把标准输出/标准错误切到 UTF-8。

    Windows 控制台默认代码页是 GBK（cp936）。脚本里的中文提示、
    上标字符（μ、²、≥）一律落到 GBK 编码器上，只要出现 GBK 未收录的
    字符（如 '²'）就会抛 UnicodeEncodeError 并让整个进程以非 0 退出——
    而此时结果文件其实已经写好了，故障发生在最后几行日志上，
    排查起来极具误导性。所有入口脚本在 main() 第一行调用本函数。

    无法重配置的环境（被重定向的管道、老版本解释器）静默跳过：
    errors="replace" 已经保证不会再因编码失败而崩溃。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def write_csv(df: pd.DataFrame, path: str | os.PathLike) -> Path:
    """统一的 CSV 落盘：自动建目录 + UTF-8 BOM（Excel 直接双击不乱码）。

    ``lineterminator="\\n"`` 是刻意为之：pandas 默认用 ``os.linesep``，
    在 Windows 上会写出 CRLF，导致同一份代码在 Windows/Linux 上产物的
    sha256 不同，MANIFEST 校验与 Git 换行符归一化都会假性报差异。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n")
    return path


def load_csv(path: str | os.PathLike) -> pd.DataFrame:
    """统一的 CSV 读取，兼容带 BOM 的文件。"""
    return pd.read_csv(path, encoding="utf-8-sig")


def make_group_splitter(n_groups: int, max_splits: int = 5) -> "GroupKFold":
    """
    构造受试者分组交叉验证器。

    GroupKFold 要求 n_splits <= 组数，否则直接抛错；亚组分析时受试者数
    可能很少，故先按组数夹紧。下限 2（少于 2 组无法做交叉验证）。
    """
    _, GroupKFold, _, _, _ = _sklearn()
    n_splits = max(2, min(int(max_splits), int(n_groups)))
    return GroupKFold(n_splits=n_splits)


def _metric(y_true, y_pred, name: str) -> float:
    """
    统一的指标计算入口（纯 numpy 实现，不依赖 sklearn）。

    自己实现而不是调 sklearn 的原因：重采样后 ``y_true`` 可能退化成常数，
    sklearn 的 ``r2_score`` 会发警告并返回 NaN，污染 bootstrap 分布；
    这里显式判掉，并把 MAE / RMSE / R² 的定义写在明处。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return float("nan")

    resid = y_pred - y_true
    if name == "mae":
        return float(np.mean(np.abs(resid)))
    if name == "rmse":
        return float(np.sqrt(np.mean(resid ** 2)))
    if name == "r2":
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        if ss_tot <= 1e-12:
            return float("nan")
        return float(1.0 - float(np.sum(resid ** 2)) / ss_tot)
    raise ValueError(f"未知指标 {name!r}，可选 {METRIC_NAMES}")


# --------------------------------------------------------------------------
# 1. 编码与分层
# --------------------------------------------------------------------------

def encode_sex(values) -> np.ndarray:
    """
    把性别列编码成 0/1（F=0, M=1）。

    为什么用 0/1 而不是 one-hot：性别是二值变量，one-hot 会产生哑变量陷阱
    （两列完全共线），与 Ridge/Lasso 的正则化放在一起会让系数不可解释。

    为什么遇错要抛异常：原代码用 ``df["sex"].map({...})``，一旦出现
    未预料写法（如 "男"）就静默变成 NaN，整行被 ``dropna`` 悄悄删掉，
    分析照样"跑通"，但样本量已经变了。这里改成显式失败。
    """
    out, bad = [], set()
    for v in values:
        if isinstance(v, str):
            key = v.strip()
        else:
            key = v
        if isinstance(key, float) and np.isnan(key):
            bad.add("NaN/空值")
            out.append(np.nan)
            continue
        if key in SEX_MAP:
            out.append(float(SEX_MAP[key]))
        else:
            bad.add(repr(v))
            out.append(np.nan)

    if bad:
        allowed = "M/F/male/female/1/0"
        raise ValueError(
            "encode_sex: 无法映射的性别取值 → " + ", ".join(sorted(bad))
            + f"。允许的取值形式：{allowed}。"
            "请先在原始数据里把这些取值改成规范形式，不要让它静默变成缺失值。"
        )
    return np.asarray(out, dtype=float)


def assign_renal_group(egfr_values, tiers: str = "all") -> pd.Series:
    """
    按 eGFR 分层。

    Parameters
    ----------
    egfr_values : array-like
    tiers : {"all", "study"}
        ``"all"``   三层：eGFR<60 / eGFR 60-89 / eGFR>=90
        ``"study"`` 两层：eGFR 60-89 / eGFR>=90；
                    eGFR<60 者返回 ``EXCLUDED_RENAL_LABEL``，由调用方过滤
                    （对齐验证方案「排除 CKD stage 3+」的入组标准）。

    Returns
    -------
    pd.Series[object]
        分层标签；eGFR 缺失处为 NaN。
    """
    egfr = pd.to_numeric(pd.Series(np.asarray(egfr_values).ravel()), errors="coerce")
    labels = []
    for v in egfr:
        if pd.isna(v):
            labels.append(np.nan)
        elif v < RENAL_BINS[1]:
            labels.append(RENAL_LABELS[0])
        elif v < RENAL_BINS[2]:
            labels.append(RENAL_LABELS[1])
        else:
            labels.append(RENAL_LABELS[2])

    out = pd.Series(labels, index=egfr.index, dtype=object)
    if tiers == "study":
        out = out.map(lambda x: EXCLUDED_RENAL_LABEL if x == RENAL_LABELS[0] else x)
    elif tiers != "all":
        raise ValueError(f"tiers 只能是 'all' 或 'study'，收到 {tiers!r}")
    return out


def dedup_columns(*groups) -> list:
    """
    把若干列名分组拼成一个**保序去重**的列表。

    为什么需要它：pandas 在 ``df[cols]`` 里若 ``cols`` 含重复标签，会返回重复的
    列；而后续 ``df[features]`` 又会**把所有同名列一并取出**，于是训练时的
    特征数（如 7）与推理时的特征数（如 6）不一致，报出
    ``X has 6 features, but StandardScaler is expecting 7 features``。
    这类错的现场在 sklearn 内部，回溯信息完全不指向真正的原因，
    因此在拼列名时就去重，比事后调试划算得多。

    用法::

        cols = dedup_columns(features, ["blood_UA", "eGFR"], ["subject_id"])
    """
    out = []
    seen = set()
    for g in groups:
        if g is None:
            continue
        items = [g] if isinstance(g, str) else list(g)
        for c in items:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


def resolve_features(features, marker_mode: str = "gold"):
    """
    把特征列表里的标记物列换成部署模式下真正能拿到的列。

    ``marker_mode="gold"``   用 ``sweat_UA``（金标准酶法测得的汗液尿酸浓度）
    ``marker_mode="sensor"`` 用 ``sweat_UA_sensor``（电极原始响应，含 pH 增益）
    """
    if marker_mode == "gold":
        return list(features)
    if marker_mode == "sensor":
        return ["sweat_UA_sensor" if f == "sweat_UA" else f for f in features]
    raise ValueError(f"marker_mode 只能是 'gold' 或 'sensor'，收到 {marker_mode!r}")


# --------------------------------------------------------------------------
# 2. 数据加载
# --------------------------------------------------------------------------

def load_merged_data(path, blood_unit: str = "umol/L", strict: bool = True):
    """
    统一加载汗液–血液配对尿酸数据。

    处理链
    ------
    列名 strip → 性别编码 → 数值列强制数值化 → 血液单位统一为 μmol/L
    → (subject_id, time_point) 唯一性校验。

    为什么统一成 μmol/L：临床指南阈值（男 420、女 360 μmol/L）和试剂盒
    说明书（线性上限 250 μmol/L）都是 μmol/L。全管线只用一种单位，
    报告里就不会出现 mg/dL 与 μmol/L 混着写的情况。

    Parameters
    ----------
    path : str
    blood_unit : {"umol/L", "mg/dL", "auto"}
        ``"auto"`` 按中位数判断（>25 视为 μmol/L）。
    strict : bool
        为 True 时遇到重复 (subject_id, time_point) 直接抛错。

    Returns
    -------
    (df, meta) : (pd.DataFrame, dict)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"数据文件不存在：{path}")

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    meta = {
        "data_path": str(path),
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else 0,
        "blood_unit": blood_unit,
        "missing_columns": [],
        "synthetic_flag": False,
        "caveat": "",
        "warnings": [],
    }

    # --- ① 性别编码（必须在数值化之前！否则 'M'/'F' 会被 to_numeric 变成 NaN）
    if "sex" in df.columns:
        col = df["sex"]
        already_binary = (
            pd.api.types.is_numeric_dtype(col)
            and col.dropna().isin([0, 1]).all()
            and col.notna().any()
        )
        if not already_binary:
            df["sex"] = encode_sex(col.tolist())

    # --- ② 数值化：除了 ID / 布尔列，其余一律转数值
    for c in df.columns:
        if c in ID_COLS or c in BOOL_COLS:
            continue
        if pd.api.types.is_bool_dtype(df[c]):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # --- ③ 血液单位统一到 μmol/L
    if "blood_UA" in df.columns:
        unit = blood_unit
        if unit == "auto":
            med = float(df["blood_UA"].median())
            unit = "mg/dL" if med < 25 else "umol/L"
            meta["blood_unit_detected"] = unit
        if unit == "mg/dL":
            df["blood_UA"] = df["blood_UA"] * BLOOD_UA_UMOL_PER_MGDL
            meta["blood_unit"] = "umol/L(由mg/dL换算)"
        elif unit == "umol/L":
            meta["blood_unit"] = "umol/L"
        else:
            raise ValueError(f"未知 blood_unit={blood_unit!r}，可选 umol/L / mg/dL / auto")

    # --- ④ 主键唯一性
    key = ["subject_id", "time_point"]
    if set(key) <= set(df.columns):
        dup = df.duplicated(subset=key, keep=False)
        if dup.any():
            msg = (f"(subject_id, time_point) 不唯一：{int(dup.sum())} 行重复，"
                   f"涉及受试者 {sorted(df.loc[dup, 'subject_id'].unique())[:8]}")
            if strict:
                raise ValueError("load_merged_data: " + msg)
            meta["warnings"].append(msg)

    # --- ⑤ 缺失列 & 合成数据标记
    meta["missing_columns"] = [c for c in EXPECTED_COLUMNS if c not in df.columns]

    meta_path = path.with_suffix("").with_suffix(".meta.json")
    if meta_path.exists():
        try:
            j = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["synthetic_flag"] = bool(j.get("synthetic", False))
            meta["caveat"] = str(j.get("caveat", ""))
            meta["seed"] = j.get("seed")
        except Exception as exc:  # noqa: BLE001 - 元数据坏了不该阻断分析
            meta["warnings"].append(f"meta.json 解析失败：{exc}")

    return df, meta


def check_design(df: pd.DataFrame, meta: dict | None = None) -> pd.DataFrame:
    """
    生成「数据质量与设计核对表」，报告附录可直接引用。

    逐列给出：数值列数、缺失数、最小值、中位数、最大值、单位；
    另附每受试者测量次数分布与关键配对覆盖率。
    """
    rows = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and c not in ID_COLS:
            rows.append({
                "column": c,
                "n": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
                "min": round(float(s.min()), 3) if s.notna().any() else np.nan,
                "median": round(float(s.median()), 3) if s.notna().any() else np.nan,
                "max": round(float(s.max()), 3) if s.notna().any() else np.nan,
                "note": "",
            })
        else:
            rows.append({
                "column": c, "n": int(s.notna().sum()),
                "n_missing": int(s.isna().sum()),
                "min": np.nan, "median": np.nan, "max": np.nan,
                "note": "非数值列",
            })

    out = pd.DataFrame(rows)

    if meta:
        extra = [
            {"column": "@n_rows", "n": meta.get("n_rows"), "note": "总行数"},
            {"column": "@n_subjects", "n": meta.get("n_subjects"), "note": "受试者数"},
            {"column": "@blood_unit", "n": None, "note": meta.get("blood_unit", "")},
            {"column": "@synthetic", "n": None,
             "note": "是（仅流程验证）" if meta.get("synthetic_flag") else "否"},
        ]
        out = pd.concat([out, pd.DataFrame(extra)], ignore_index=True)

    if "sweat_UA_censored" in df.columns:
        rate = float(df["sweat_UA_censored"].mean() * 100)
        out = pd.concat([out, pd.DataFrame([{
            "column": "@censoring_rate_pct", "n": round(rate, 2),
            "note": "汗液 UA 低于试剂盒 LOD(8 μmol/L) 的比例",
        }])], ignore_index=True)

    return out


# --------------------------------------------------------------------------
# 3. 无泄漏交叉验证
# --------------------------------------------------------------------------

def _split(splitter, X, y, groups):
    """
    统一的折划分入口。

    ``KFold`` 会忽略 ``groups`` 并发出 ``UserWarning: The groups parameter
    is ignored by KFold``；``GroupKFold`` 则必须有 ``groups``。为了用一个调用
    同时覆盖两者，这里照常传 ``groups``，但只把这一条已知警告静音——
    它出现在「故意制造泄漏的对照组」里属于预期行为，
    留着会淹没真正需要注意的警告。
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*groups parameter is ignored.*", category=UserWarning)
        yield from splitter.split(X, y, groups=groups)


def oof_predict(df, features, estimator, splitter,
                target: str = "blood_UA",
                group_col: str = "subject_id",
                time_col: str = "time_point",
                allow_subject_overlap: bool = False):
    """
    折外（out-of-fold, OOF）预测。

    关键点：标准化器放进 ``Pipeline``，每折在训练部分 ``fit``、在测试部分
    ``transform``。原实现先对全量 X 做 ``fit_transform`` 再切折，测试折的
    均值/方差已经漏进训练折，是典型的 CV 泄漏。

    ``allow_subject_overlap`` 默认为 False，即检测到「同一受试者跨折」直接抛错
    ——同一人的多次测量高度相关，跨折会让模型变相"背过"该受试者，折外误差被
    系统性高估，这是本项目最要防的一类漏洞。唯一需要置 True 的场景，是
    ``kfold_optimism_delta()`` 里**故意**用随机 KFold 制造泄漏当对照组，
    用来量化"不按受试者分组会乐观多少"。除此以外任何地方打开它都是缺陷。

    Returns
    -------
    (oof_df, fold_info)
        oof_df  : subject_id, time_point, y_true, y_pred, fold
        fold_info: list[dict(fold, n_train, n_test, train_subject_ids, test_subject_ids])
    """
    cols = dedup_columns([group_col, time_col, target], features)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"oof_predict: 数据缺少列 {missing}")

    sub = df[cols].dropna().reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError("oof_predict: 去掉缺失后没有剩余样本")

    clone, _, _, Pipeline, StandardScaler = _sklearn()

    X = sub[list(features)].to_numpy(dtype=float)
    y = sub[target].to_numpy(dtype=float)
    g = sub[group_col].to_numpy()

    oof = np.full(len(sub), np.nan)
    fold_ids = np.full(len(sub), -1, dtype=int)
    fold_info = []

    for k, (tr, te) in enumerate(_split(splitter, X, y, g)):
        pipe = Pipeline([("scaler", StandardScaler()), ("model", clone(estimator))])
        pipe.fit(X[tr], y[tr])
        oof[te] = pipe.predict(X[te])
        fold_ids[te] = k
        train_ids = np.unique(g[tr]).tolist()
        test_ids = np.unique(g[te]).tolist()
        overlap = sorted(set(train_ids) & set(test_ids))
        fold_info.append({
            "fold": k,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "n_train_subjects": len(train_ids),
            "n_test_subjects": len(test_ids),
            "train_subject_ids": train_ids,
            "test_subject_ids": test_ids,
            "subject_overlap": overlap,
        })

    if np.isnan(oof).any():
        raise RuntimeError(
            f"oof_predict: 有 {int(np.isnan(oof).sum())} 行没有得到折外预测，"
            "说明 splitter 没有覆盖全部样本，请检查折划分。"
        )
    if any(f["subject_overlap"] for f in fold_info) and not allow_subject_overlap:
        leaked = sorted({i for f in fold_info for i in f["subject_overlap"]})
        raise RuntimeError(
            f"oof_predict: 检测到受试者同时出现在训练折与测试折（数据泄漏），"
            f"共 {len(leaked)} 人：{leaked[:10]}"
            f"{' ...' if len(leaked) > 10 else ''}。"
            "若这是为了量化泄漏影响而故意为之，请显式传 allow_subject_overlap=True。"
        )

    oof_df = sub[[group_col, time_col]].copy()
    oof_df["y_true"] = y
    oof_df["y_pred"] = oof
    oof_df["fold"] = fold_ids
    return oof_df, fold_info


def summarize_oof(oof_df, name: str = "", group_col: str = "subject_id") -> dict:
    """把一份 OOF 预测汇总成 MAE / R² / RMSE（整体口径）。"""
    yt, yp = oof_df["y_true"].to_numpy(), oof_df["y_pred"].to_numpy()
    return {
        "name": name,
        "n_used": int(len(oof_df)),
        "n_subjects_used": int(oof_df[group_col].nunique()),
        "mae": _metric(yt, yp, "mae"),
        "r2": _metric(yt, yp, "r2"),
        "rmse": _metric(yt, yp, "rmse"),
    }


# --------------------------------------------------------------------------
# 4. 不确定度
# --------------------------------------------------------------------------

def cluster_bootstrap_ci(oof_df, metric: str = "mae", n_boot: int = 1000,
                         seed: int = 42, alpha: float = 0.05,
                         group_col: str = "subject_id") -> dict:
    """
    受试者簇 bootstrap 置信区间。

    为什么按受试者整簇重采样：同一受试者的多次测量高度相关，
    按测量条重采样相当于假装样本量是「测量条数」，
    会系统性地把置信区间压得过窄（伪重复 / pseudoreplication）。

    Returns
    -------
    dict(metric, point, ci_low, ci_high, n_boot, n_subjects, seed)
    """
    if metric not in METRIC_NAMES:
        raise ValueError(f"未知指标 {metric!r}，可选 {METRIC_NAMES}")

    groups = oof_df[group_col].to_numpy()
    uniq = np.unique(groups)
    idx_by_group = {u: np.where(groups == u)[0] for u in uniq}
    yt_all = oof_df["y_true"].to_numpy(dtype=float)
    yp_all = oof_df["y_pred"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for b in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[p] for p in pick])
        stats[b] = _metric(yt_all[idx], yp_all[idx], metric)

    stats = stats[~np.isnan(stats)]
    if stats.size == 0:
        lo = hi = float("nan")
    else:
        lo = float(np.percentile(stats, 100 * alpha / 2))
        hi = float(np.percentile(stats, 100 * (1 - alpha / 2)))

    return {
        "metric": metric,
        "point": _metric(yt_all, yp_all, metric),
        "ci_low": lo,
        "ci_high": hi,
        "n_boot": int(n_boot),
        "n_subjects": int(len(uniq)),
        "seed": int(seed),
    }


def paired_delta_ci(oof_a, oof_b, metric: str = "r2", n_boot: int = 1000,
                    seed: int = 42, alpha: float = 0.05,
                    group_col: str = "subject_id"):
    """
    配对差值 bootstrap：Δ = metric(A) − metric(B)。

    合法性前提：``GroupKFold`` 不做 shuffle，折划分是确定性的，
    因此所有模型共享**同一套折**，逐条 OOF 预测可以合法配对。
    （若换成带 shuffle 的划分，各模型折不同，配对就失去意义。）

    Returns
    -------
    (merged_df, summary_dict)
    """
    a = oof_a.rename(columns={"y_pred": "y_pred_a"})[
        [group_col, "time_point", "y_true", "y_pred_a"]]
    b = oof_b.rename(columns={"y_pred": "y_pred_b"})[
        [group_col, "time_point", "y_pred_b"]]
    m = a.merge(b, on=[group_col, "time_point"], how="inner")
    if len(m) == 0:
        raise ValueError("paired_delta_ci: 两个模型的 OOF 预测没有公共样本")

    groups = m[group_col].to_numpy()
    uniq = np.unique(groups)
    idx_by_group = {u: np.where(groups == u)[0] for u in uniq}
    yt = m["y_true"].to_numpy(dtype=float)
    ya = m["y_pred_a"].to_numpy(dtype=float)
    yb = m["y_pred_b"].to_numpy(dtype=float)

    rng = np.random.default_rng(seed)
    deltas = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[p] for p in pick])
        deltas[k] = (_metric(yt[idx], ya[idx], metric)
                     - _metric(yt[idx], yb[idx], metric))

    deltas = deltas[~np.isnan(deltas)]
    point = _metric(yt, ya, metric) - _metric(yt, yb, metric)
    if deltas.size == 0:
        lo = hi = float("nan")
        p_better = float("nan")
    else:
        lo = float(np.percentile(deltas, 100 * alpha / 2))
        hi = float(np.percentile(deltas, 100 * (1 - alpha / 2)))
        # 「A 优于 B」的比例：MAE 越小越好，R²/RMSE 越大越好
        better = deltas < 0 if metric in ("mae", "rmse") else deltas > 0
        p_better = float(better.mean())

    summary = {
        "metric": metric,
        "point_delta": point,
        "ci_low": lo,
        "ci_high": hi,
        "p_A_better": p_better,
        # 「区间不跨 0」等价于双侧 α 水平上差值显著
        "ci_excludes_zero": bool((lo > 0 and hi > 0) or (lo < 0 and hi < 0))
        if not (np.isnan(lo) or np.isnan(hi)) else False,
        "n_boot": int(n_boot),
        "n_paired": int(len(m)),
        "n_subjects": int(len(uniq)),
        "seed": int(seed),
    }
    return m, summary


def kfold_optimism_delta(df, features, estimator, target: str = "blood_UA",
                         group_col: str = "subject_id", n_splits: int = 5,
                         seed: int = 42, metric: str = "r2",
                         n_boot: int = 1000):
    """
    量化「随机 KFold」相对「受试者分组 GroupKFold」的乐观偏差。

    同一受试者的多次测量如果同时出现在训练折与测试折，模型可以靠
    「记住这个人」而不是「学会这个规律」拿到高分。这里用同一特征集、
    同一估计器，只改折划分方式，直接比较两者的 OOF 表现。

    Returns
    -------
    dict(group=..., shuffle=..., delta_r2_vs_group=...)
    """
    g = df[group_col].nunique()
    _, _, KFold, _, _ = _sklearn()
    oof_group, _ = oof_predict(
        df, features, estimator,
        make_group_splitter(g, n_splits), target=target, group_col=group_col)
    oof_shuf, _ = oof_predict(
        df, features, estimator,
        KFold(n_splits=max(2, min(n_splits, len(df) // 2)),
             shuffle=True, random_state=seed),
        target=target, group_col=group_col,
        # 这里的泄漏是**故意**制造的对照组：本函数的目的正是量化
        # 「不按受试者分组会乐观多少」，所以必须放行受试者跨折。
        allow_subject_overlap=True)

    res_g = cluster_bootstrap_ci(oof_group, metric=metric, n_boot=n_boot, seed=seed)
    res_s = cluster_bootstrap_ci(oof_shuf, metric=metric, n_boot=n_boot, seed=seed)
    _, dlt = paired_delta_ci(oof_shuf, oof_group, metric=metric,
                             n_boot=n_boot, seed=seed)

    return {
        "group_point": res_g["point"],
        "group_ci_low": res_g["ci_low"],
        "group_ci_high": res_g["ci_high"],
        "shuffle_point": res_s["point"],
        "shuffle_ci_low": res_s["ci_low"],
        "shuffle_ci_high": res_s["ci_high"],
        "delta_shuffle_vs_group": dlt["point_delta"],
        "delta_ci_low": dlt["ci_low"],
        "delta_ci_high": dlt["ci_high"],
        "delta_ci_excludes_zero": dlt["ci_excludes_zero"],
        "metric": metric,
        "n_boot": int(n_boot),
        "seed": int(seed),
    }


__all__ = [
    "BLOOD_UA_UMOL_PER_MGDL", "SEX_MAP", "RENAL_BINS", "RENAL_LABELS",
    "RENAL_GROUP_ORDER", "STUDY_RENAL_LABELS", "EXCLUDED_RENAL_LABEL",
    "EXPECTED_COLUMNS", "METRIC_NAMES",
    "write_csv", "load_csv", "make_group_splitter", "encode_sex",
    "assign_renal_group", "resolve_features", "load_merged_data",
    "check_design", "oof_predict", "summarize_oof", "cluster_bootstrap_ci",
    "paired_delta_ci", "kfold_optimism_delta", "setup_console",
    "dedup_columns",
]
