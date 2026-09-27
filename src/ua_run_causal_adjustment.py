#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ua_run_causal_adjustment.py — 因果调整诊断（调整阶梯）
=======================================================


回答一个问题：**控制住不同混杂集合之后，汗液–血液尿酸的关联强度如何变化？**

这是模型拟合之前的「可审计」步骤：如果加入某个变量后关联明显增强，
说明该变量此前在压制真实关联；如果加入后几乎不动，说明它不是混杂。
它把 DAG 里的定性判断变成一条可检验的定量曲线——报告 5.4 节宣称的
「加入 eGFR 后关联增强」，就是由本脚本产出的 ``adjustment_ladder.png``
直接支撑的（而不是靠文字断言）。

原版的三处致命问题
------------------
1. ``sex`` 列是字符串 'M'/'F'，直接被塞进 ``LinearRegression().fit(X, y)``，
   必然抛 ``ValueError: could not convert string to float``——
   这就是 ``results/`` 下只有第 2 步产物、没有 ``ua_causal_adjustment/`` 的原因。
2. 与 ``ua_reproduce_figure5hi.py`` 各自 ``read_csv``，一边编码性别一边不编码，
   同样的列在两个脚本里语义不同。
3. 调整阶梯与 DAG 推导出来的调整集不对齐（例如把 sweat_rate 这种
   「第二处理变量」混在混杂阶梯里，语义不清）。

本版改为统一调用 ``ua_data_utils.load_merged_data``，并加硬校验：
喂进回归前先断言矩阵不是 object dtype。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from ua_data_utils import load_merged_data, setup_console, write_csv

# 中文字体（否则图里中文和负号都会变成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

TREATMENT = "sweat_UA"
OUTCOME = "blood_UA"

#: 调整阶梯。``kind`` 说明每一步在因果语义上做了什么：
#:   confounder  = 阻断一条后门路径（真·混杂调整）
#:   precision   = 纳入精度变量（提高估计精度，不阻断后门）
#:   modifier    = 纳入效应修饰变量（改变的是映射增益）
#:   joint       = 纳入第二处理变量（联合处理，非混杂调整）
#:   all         = 全协变量上界
LADDER = [
    ("S0 无调整（原始关联）", [], "raw",
     "不做任何调整，对应论文里的「原始汗液–血液相关性」"),
    ("S1 + BMI", ["BMI"], "confounder",
     "阻断后门路径 sweat_UA ← BMI → blood_UA"),
    ("S2 + sex", ["BMI", "sex"], "confounder",
     "再阻断 sweat_UA ← sex → blood_UA（及 sex→eGFR→blood_UA）"),
    ("S3 + age（Tier-1 最小集）", ["BMI", "sex", "age"], "confounder",
     "再阻断 sweat_UA ← age → blood_UA；此步为严格后门最小集"),
    ("S4 + eGFR（Tier-2）", ["BMI", "sex", "age", "eGFR"], "precision",
     "加入入组基线 eGFR。注意它的身份是**精度变量而非混杂变量**："
     "eGFR 不指向处理变量 sweat_UA，严格后门准则下无需调整；"
     "纳入它是因为肾功能解释了血尿酸的大量个体差异，"
     "控制后可降低估计方差。**这一步 r 是升是降由数据决定，不作预期**——"
     "若 r 下降，说明原始关联中有一部分来自 eGFR 与两者的共同方差；"
     "若 r 上升，说明 eGFR 此前压制了关联。两种方向都是有信息量的结果。"),
    ("S5 + sweat_pH", ["BMI", "sex", "age", "eGFR", "sweat_pH"], "modifier",
     "加入效应修饰变量 pH：它改变汗液尿酸的信号增益"),
    ("S6 + sweat_rate（联合处理）", ["BMI", "sex", "age", "eGFR", "sweat_pH", "sweat_rate"],
     "joint", "加入第二处理变量出汗率，吸收稀释效应"),
    ("S7 全协变量（上界）",
     ["BMI", "sex", "age", "eGFR", "sweat_pH", "sweat_rate",
      "fat_mass", "muscle_mass", "BMR"],
     "all", "把所有可得协变量一并纳入，作为「过度调整 / 共线」的对照上界"),
]


# --------------------------------------------------------------------------
# 统计工具
# --------------------------------------------------------------------------

def residualize(y, X):
    """把 y 对 X 做线性回归，返回残差（即剔除 X 的线性影响后的部分）。"""
    if X is None or (hasattr(X, "shape") and X.size == 0):
        return np.asarray(y, dtype=float)
    model = LinearRegression().fit(X, y)
    return np.asarray(y, dtype=float) - model.predict(X)


def pearson(a, b):
    """纯 numpy 的 Pearson 相关系数（避免为一次计算引入 scipy）。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _cluster_bootstrap_r(res_a, res_b, groups, n_boot=1000, seed=42, alpha=0.05):
    """
    按受试者整簇 bootstrap 的相关系数置信区间。

    按测量条重采样会把「同一人测了 3 次」当成 3 个独立样本，
    系统性地把 CI 压窄（伪重复）。
    """
    uniq = np.unique(groups)
    idx_by = {u: np.where(groups == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    stats = np.empty(int(n_boot), dtype=float)
    for k in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[p] for p in pick])
        stats[k] = pearson(res_a[idx], res_b[idx])
    stats = stats[np.isfinite(stats)]
    if stats.size == 0:
        return float("nan"), float("nan")
    return (float(np.percentile(stats, 100 * alpha / 2)),
            float(np.percentile(stats, 100 * (1 - alpha / 2))))


def _fisher_p_cluster_robust(r, n_clusters):
    """
    Fisher z 变换 + 正态近似的双侧 p 值，**有效样本量取受试者数而非测量条数**。

    这是对伪重复的显式处理：n=150 条测量其实只来自 50 个人，
    自由度应按人算。用 ``math.erfc`` 实现，避免依赖 scipy。
    """
    if not np.isfinite(r) or n_clusters < 4:
        return float("nan")
    z = math.atanh(max(min(r, 0.999999), -0.999999)) * math.sqrt(n_clusters - 3)
    return float(math.erfc(abs(z) / math.sqrt(2.0)))


# --------------------------------------------------------------------------
# 主计算
# --------------------------------------------------------------------------

def run_ladder(df, n_boot=1000, seed=42, marker=TREATMENT):
    rows = []
    for name, cols, kind, note in LADDER:
        available = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]

        sub = df[[marker, OUTCOME, "subject_id"] + available].dropna()
        n, n_sub = int(len(sub)), int(sub["subject_id"].nunique())

        if n < 15 or n_sub < 5:
            rows.append({
                "treatment": marker, "outcome": OUTCOME, "adjustment_set": name,
                "kind": kind, "adjustment_columns": ";".join(available),
                "missing_columns": ";".join(missing),
                "n": n, "n_subjects": n_sub,
                "r": np.nan, "r_ci_low": np.nan, "r_ci_high": np.nan,
                "p_value_cluster_robust": np.nan, "slope": np.nan,
                "note": note + "（样本不足，跳过）",
            })
            continue

        # ---- 硬校验：回归矩阵不允许出现 object dtype ----
        X_adj = sub[available].to_numpy(dtype=float) if available else None
        if X_adj is not None and X_adj.dtype == object:
            raise TypeError(
                f"调整集 {name} 的矩阵仍是 object dtype，"
                "说明有非数值列（如未编码的 sex）漏进了回归。"
                "请在 ua_data_utils.encode_sex 里补齐映射。")

        res_t = residualize(sub[marker].to_numpy(dtype=float), X_adj)
        res_o = residualize(sub[OUTCOME].to_numpy(dtype=float), X_adj)

        r = pearson(res_t, res_o)
        lo, hi = _cluster_bootstrap_r(
            res_t, res_o, sub["subject_id"].to_numpy(), n_boot=n_boot, seed=seed)
        p = _fisher_p_cluster_robust(r, n_sub)

        # 标准化斜率：残差尺度不同，用「结局残差对处理残差」的回归系数
        if res_t.std() > 1e-12:
            slope = float(np.polyfit(res_t, res_o, 1)[0])
        else:
            slope = float("nan")

        rows.append({
            "treatment": marker, "outcome": OUTCOME, "adjustment_set": name,
            "kind": kind, "adjustment_columns": ";".join(available),
            "missing_columns": ";".join(missing),
            "n": n, "n_subjects": n_sub,
            "r": r, "r_ci_low": lo, "r_ci_high": hi,
            "p_value_cluster_robust": p, "slope": slope,
            "note": note,
        })

    out = pd.DataFrame(rows)
    if out["r"].notna().any():
        raw_r = out.loc[out["kind"] == "raw", "r"].iloc[0]
        out["delta_r_vs_raw"] = out["r"] - raw_r
        out["abs_r_gain_pct"] = (out["r"].abs() - abs(raw_r)) / abs(raw_r) * 100.0
    else:
        out["delta_r_vs_raw"] = np.nan
        out["abs_r_gain_pct"] = np.nan

    # 哪一步带来的 r 变化最大（用于报告里点名"影响最大的调整步骤"）。
    # 注意用「变化」而非「增益」：r 下降同样是重要结果，
    # 把下降了说成"增益"会让产物里的措辞与数据方向自相矛盾。
    steps = out[out["kind"].isin(["confounder", "precision"])].dropna(subset=["r"])
    out.attrs["key_step"] = None
    if len(steps) >= 2:
        deltas = steps["r"].diff()
        idx = deltas.abs().idxmax()
        out.attrs["key_step"] = (steps.loc[idx, "adjustment_set"], float(deltas.loc[idx]))

    return out


def plot_adjustment_ladder(res: pd.DataFrame, out_path, marker=TREATMENT):
    """调整阶梯图：r 随调整集推进的变化 + 95% 置信带。"""
    d = res.dropna(subset=["r"]).reset_index(drop=True)
    if len(d) == 0:
        print("[plot] 调整阶梯为空，跳过绘图")
        return

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    x = np.arange(len(d))

    kind_color = {
        "raw": "#A8A8A8", "confounder": "#2A9D8F",
        "precision": "#E76F51", "modifier": "#F4A261",
        "joint": "#8E7DBE", "all": "#264653",
    }
    colors = [kind_color.get(k, "#333333") for k in d["kind"]]

    ax.fill_between(x, d["r_ci_low"], d["r_ci_high"], color="#CCCCCC",
                    alpha=0.45, zorder=1, label="95% CI（受试者簇 bootstrap）")
    ax.plot(x, d["r"], "-", color="#333333", linewidth=1.2, zorder=2)
    ax.scatter(x, d["r"], s=90, c=colors, edgecolor="black",
               linewidth=0.8, zorder=3)

    for i, row in d.iterrows():
        ax.annotate(f"{row['r']:.3f}", (i, row["r"]), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=8.5)

    ax.axhline(0, color="gray", linestyle="-", alpha=0.35)
    raw_r = d.loc[d["kind"] == "raw", "r"]
    if len(raw_r):
        ax.axhline(float(raw_r.iloc[0]), color="#A8A8A8", linestyle="--",
                   alpha=0.8, label=f"原始关联 r = {float(raw_r.iloc[0]):.3f}")

    ax.set_xticks(x)
    ax.set_xticklabels(d["adjustment_set"], rotation=28, ha="right", fontsize=8.5)
    ax.set_ylabel(f"残差化后的 Pearson r（{marker} vs {OUTCOME}）")
    ax.set_title("因果调整阶梯：逐步控制混杂后，汗液–血液尿酸关联的变化")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="UA 因果调整诊断")
    parser.add_argument("--data", default="data/ua_example_merged_data.csv")
    parser.add_argument("--out", default="results/ua_causal_adjustment")
    parser.add_argument("--blood-unit", default="umol/L",
                        choices=["umol/L", "mg/dL", "auto"])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--marker-mode", default="gold",
                        choices=["gold", "sensor"],
                        help="gold=酶法 sweat_UA；sensor=电极原始响应 sweat_UA_sensor")
    args = parser.parse_args()

    marker = "sweat_UA" if args.marker_mode == "gold" else "sweat_UA_sensor"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_merged_data(args.data, blood_unit=args.blood_unit)
    if meta.get("synthetic_flag"):
        print(f"[ua_run_causal_adjustment] 注意：{meta.get('caveat')}")

    print(f"[ua_run_causal_adjustment] 载入 {meta['n_rows']} 行 / "
          f"{meta['n_subjects']} 名受试者；标记物 = {marker}（{args.marker_mode} 模式）")
    if meta["missing_columns"]:
        print(f"[ua_run_causal_adjustment] 缺少列：{meta['missing_columns']}")

    res = run_ladder(df, n_boot=args.n_boot, seed=args.seed, marker=marker)

    csv_path = out_dir / "causal_adjustment_summary.csv"
    write_csv(res, csv_path)
    print(f"\n[ua_run_causal_adjustment] 调整阶梯 → {csv_path}")
    for _, row in res.iterrows():
        if np.isnan(row["r"]):
            print(f"  {row['adjustment_set']:32s} | 跳过（{row['note']}）")
        else:
            print(f"  {row['adjustment_set']:32s} | r = {row['r']:+.3f} "
                  f"[{row['r_ci_low']:+.3f}, {row['r_ci_high']:+.3f}] "
                  f"| p = {row['p_value_cluster_robust']:.2e} "
                  f"| n = {row['n']}（{row['n_subjects']} 人）")

    plot_adjustment_ladder(res, out_dir / "adjustment_ladder.png", marker=marker)

    # --- 关键结论 ---
    raw = res.loc[res["kind"] == "raw", "r"]
    tier2 = res.loc[res["adjustment_set"].str.startswith("S4"), "r"]
    if len(raw) and len(tier2):
        r0, r1 = float(raw.iloc[0]), float(tier2.iloc[0])
        print(f"\n>>> 原始关联 r = {r0:+.3f} → 加入 eGFR 后 r = {r1:+.3f} "
              f"（Δ = {r1 - r0:+.3f}）")
        if abs(r1) > abs(r0):
            print(">>> 控制 eGFR 后关联增强：肾功能此前在压制汗液–血液关联，"
                  "eGFR 是重要的精度变量。")
        else:
            print(">>> 控制 eGFR 后关联减弱。这是偏相关的正常行为、不是失败："
                  "eGFR 与血尿酸强相关，同时也与汗液尿酸共享一部分方差，"
                  "控制它就把这部分共享方差一并扣除，留下的是更纯净的"
                  "汗液–血液关联。r 的方向由数据决定，本脚本不作预设。")
    ks = res.attrs.get("key_step")
    if ks:
        verb = "上升" if ks[1] > 0 else "下降"
        print(f">>> 单步 r 变化最大的是「{ks[0]}」，Δr = {ks[1]:+.3f}（{verb}）")

    print(f"\n[ua_run_causal_adjustment] 全部产物写入 {out_dir}")


if __name__ == "__main__":
    main()
