#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ua_make_example_data.py — 尿酸合成数据生成器（仅用于流程验证）
================================================================

来源与改动
----------
本文件改编自论文官方公开代码仓库 ``SIJIEJI/polycore-lipid-causal`` 的 ``src/make_example_data.py``
（检测场景由脂质迁移为尿酸，结构与参数均有结构性改动，非照搬）。
详见 README「来源与署名」与 docs/UA_MIGRATION_GUIDE.md。


**本文件生成的是合成数据，不是实验结果。**
产物同名目录下会写出 ``*.meta.json``，标记 ``"synthetic": true``；
下游脚本读取后会在报告图表上强制打印「合成数据」水印。

为什么必须重写
--------------
原版生成器里有一条**确定性反演公式**：

    sweat_ua = blood_ua * 59.5 * U(0.06, 0.14) / (1 + 0.3*(SR-0.8)) * ...

即先有 blood_ua，再按公式"算出" sweat_ua。这样造出来的数据里，
汗液尿酸几乎是血尿酸的函数，任何模型都能拿到接近 1 的 R²——
**这种数字一旦被当成性能证据，性质就是数据造假**。
（竞赛规则第 5 页明确把「伪造篡改数据、夸大成果」列为可取消资格项。）

本版改为：只在生成器里**编码生理假设**，不设定结果目标。
每条测量都经过「个体随机效应 + 逐次测量噪声 + 比例/绝对双底噪」三层扰动，
因此最终的相关强度是这些假设的**推论**，而不是预设值。

编码的生理假设
--------------
1. 血尿酸由肾功能（eGFR）主导（约 70% 经肾排泄），BMI、性别、年龄次之，
   并含一个**大幅度的个体随机效应**（人群中血尿酸 SD ≈ 75 μmol/L，
   这一部分任何静态协变量都预测不了，只能靠实际测量）。
2. eGFR **由肌酐经 CKD-EPI 2021 公式导出**（而非独立抽样），保证数据自洽：
   age/sex/creatinine → eGFR → blood_UA 的因果链真实存在。
3. 汗液尿酸/血尿酸存在**个体特有的分配系数**（对数正态，几何均值 10%，
   对数 SD 0.35），即每个人"漏进汗里"的比例不同且差距可观。
4. 出汗率稀释：出汗率越高，汗液尿酸浓度越低。
5. 汗液 pH 是**效应修饰变量**：它改变尿酸的离子化比例与电极响应增益，
   但不直接决定血尿酸。
6. 电极原始响应 ``sweat_UA_sensor`` 是汗液尿酸的**含 pH 增益的观测**，
   与酶法金标准 ``sweat_UA`` 区分开——部署模式下只有前者可得。
7. 低于试剂盒检测限（LOD = 8 μmol/L）的样本按左删失处理，并置布尔标记列。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ua_data_utils import setup_console  # noqa: E402

# --------------------------------------------------------------------------
# 常量（与验证方案 / 试剂盒说明书对齐）
# --------------------------------------------------------------------------

#: 采样阶段：T0 静息基线 → T1 刺激后（温水淋浴/轻运动）→ T2 恢复 30 min
STAGES = [("T0", 0), ("T1", 40), ("T2", 70)]

#: 试剂盒检测限（BOXBIO AKAO014M 说明书：LOD 8 μmol/L，线性 25–250 μmol/L）
KIT_LOD_UMOL = 8.0

#: 血尿酸分子量与换算（mg/dL → μmol/L）
UA_MGDL_TO_UMOL = 59.48

CAVEAT = "合成数据——仅用于流程验证与代码自检，不构成任何性能证据。"


# --------------------------------------------------------------------------
# CKD-EPI 2021（race-free）肾功能估算
# --------------------------------------------------------------------------

def ckd_epi_2021(creatinine_umol, age, is_male):
    """
    CKD-EPI 2021 无种族系数公式：
        eGFR = 142 × min(Scr/κ, 1)^α × max(Scr/κ, 1)^(-1.200)
               × 0.9938^age × (1.012 if female)
    Scr 单位 mg/dL，κ = 0.7(女)/0.9(男)，α = -0.241(女)/-0.302(男)。
    """
    scr = np.asarray(creatinine_umol, dtype=float) / 88.4
    kappa = 0.9 if is_male else 0.7
    alpha = -0.302 if is_male else -0.241

    ratio = scr / kappa
    egfr = (142.0
            * np.minimum(ratio, 1.0) ** alpha
            * np.maximum(ratio, 1.0) ** (-1.200)
            * 0.9938 ** float(age))
    if not is_male:
        egfr *= 1.012
    return float(egfr)


# --------------------------------------------------------------------------
# 主生成逻辑
# --------------------------------------------------------------------------

def generate_synthetic_data(n_subjects=50, n_timepoints=3, random_seed=42):
    """
    生成汗液–血液配对尿酸合成数据。

    Parameters
    ----------
    n_subjects : int
        受试者数。默认 50，与验证方案的目标样本量一致。
    n_timepoints : int
        每人的采样点数量（最多 3，对应 T0/T1/T2）。
    random_seed : int

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(random_seed)
    stages = STAGES[:max(1, min(int(n_timepoints), len(STAGES)))]
    rows = []

    for sid in range(1, int(n_subjects) + 1):
        # ---------------- 静态人口学与体成分 ----------------
        is_male = bool(rng.random() < 0.5)
        sex = 1 if is_male else 0
        age = float(rng.uniform(30.0, 68.0))

        bmi = float(np.clip(rng.normal(26.0, 4.0), 18.5, 36.0))
        height_cm = float(rng.normal(172.0, 6.5) if is_male else rng.normal(160.0, 5.5))
        weight_kg = bmi * (height_cm / 100.0) ** 2

        # 体脂率随 BMI 上升，女性整体更高；肌量为去脂体重的 55%
        body_fat_frac = 0.18 + 0.010 * (bmi - 22.0) + (0.06 if not is_male else 0.0)
        body_fat_frac = float(np.clip(rng.normal(body_fat_frac, 0.045), 0.08, 0.48))
        fat_mass = weight_kg * body_fat_frac
        muscle_mass = (weight_kg - fat_mass) * 0.55

        # Mifflin-St Jeor 基础代谢率（与体重/身高/年龄/性别自洽）
        bmr = (10 * weight_kg + 6.25 * height_cm - 5 * age
               + (5.0 if is_male else -161.0)) + float(rng.normal(0, 60))

        # ---------------- 肾功能（肌酐 → CKD-EPI → eGFR） ----------------
        creatinine = float(np.clip(rng.normal(78.0, 0.26 * 78.0), 40.0, 230.0))
        egfr = ckd_epi_2021(creatinine, age, is_male)
        egfr = float(np.clip(egfr, 25.0, 145.0))
        bun = float(np.clip(12.0 * (creatinine / 88.4) + rng.normal(0, 2.6), 5.0, 45.0))

        blood_glucose = float(np.clip(
            rng.normal(88.0 + 0.9 * (bmi - 22.0) + 0.15 * (age - 40.0), 9.0),
            65.0, 180.0))

        # ---------------- 血尿酸基线 ----------------
        # 280: 人群基线；+80 男性；BMI 效应 5 μmol/L per unit；
        # eGFR 效应 1.6 μmol/L per unit；年龄效应 1.0 μmol/L per year；
        # 个体随机效应 SD=75 μmol/L（这一项是"静态协变量预测不了"的部分）
        blood_ua_base = (
            280.0
            + 80.0 * sex
            + 5.0 * (bmi - 22.0)
            + 1.6 * (100.0 - egfr)
            + 1.0 * (age - 40.0)
            + float(rng.normal(0, 75.0))
        )
        blood_ua_base = float(np.clip(blood_ua_base, 130.0, 700.0))

        # 嘌呤负荷反应幅度（个体差异大：有人是"反应者"，有人几乎无反应）
        purine_amp = float(max(0.0, rng.normal(35.0, 14.0)))

        # ---------------- 个体汗/血分配系数（对数正态随机效应） ----------------
        # 几何均值 10%（汗液尿酸约为血尿酸的 10%），个体间对数 SD 0.35
        ratio_i = float(np.exp(rng.normal(np.log(0.10), 0.35)))

        # ---------------- 时间序列 ----------------
        for stage_name, t_min in stages:
            # 嘌呤负荷时序：摄入后 40–70 min 内上升
            purine = purine_amp * (1.0 - np.exp(-t_min / 30.0)) * np.exp(-t_min / 260.0)
            blood_ua = float(np.clip(
                blood_ua_base + purine + rng.normal(0, 8.0), 120.0, 750.0))

            # 出汗率：T1 刺激期显著升高，T2 部分回落
            if stage_name == "T0":
                sweat_rate = float(np.clip(rng.normal(0.55, 0.15), 0.25, 1.0))
            elif stage_name == "T1":
                sweat_rate = float(np.clip(rng.normal(1.35, 0.35), 0.50, 2.4))
            else:
                sweat_rate = float(np.clip(rng.normal(0.75, 0.20), 0.30, 1.4))

            # 汗液 pH：出汗率升高时略偏酸
            sweat_ph = float(np.clip(
                rng.normal(5.9, 0.35) - 0.25 * (sweat_rate - 0.8), 4.5, 7.0))

            # ---------------- 汗液尿酸（三层扰动） ----------------
            # ① 系统部分：血尿酸 × 个体分配系数 ÷ 稀释因子 × pH 电离因子
            dilution = 1.0 + 0.30 * (sweat_rate - 0.8)
            ph_factor = 1.0 + 0.10 * (sweat_ph - 5.8)
            expected = blood_ua * ratio_i * ph_factor / dilution
            # ② 逐次测量的乘性噪声（CV≈22%）③ 加性底噪
            sweat_ua = expected * float(np.exp(rng.normal(0, 0.22))) + float(rng.normal(0, 2.0))
            sweat_ua = float(max(sweat_ua, 0.5))

            # ④ 试剂盒 LOD 左删失
            censored = bool(sweat_ua < KIT_LOD_UMOL)
            sweat_ua = float(max(sweat_ua, KIT_LOD_UMOL))

            # ---------------- 电极原始响应（部署模式唯一可得的输入） ----------------
            # pH 改变质子化状态 → 响应增益；叠加 5% 乘性 + 1.5 μmol/L 加性仪器噪声
            sensor = (sweat_ua
                      * (1.0 + 0.18 * (sweat_ph - 5.8))
                      * float(np.exp(rng.normal(0, 0.05)))
                      + float(rng.normal(0, 1.5)))
            sensor = float(max(sensor, 0.5))

            rows.append({
                "subject_id": sid,
                "stage": stage_name,
                "time_point": t_min,
                "sweat_UA": round(sweat_ua, 2),
                "sweat_UA_sensor": round(sensor, 2),
                "sweat_UA_censored": censored,
                "sweat_rate": round(sweat_rate, 3),
                "sweat_pH": round(sweat_ph, 2),
                "BMI": round(bmi, 2),
                "sex": "M" if is_male else "F",
                "age": round(age, 1),
                "eGFR": round(egfr, 1),
                "creatinine": round(creatinine, 1),
                "BUN": round(bun, 2),
                "fat_mass": round(fat_mass, 2),
                "muscle_mass": round(muscle_mass, 2),
                "BMR": round(bmr, 1),
                "blood_glucose": round(blood_glucose, 1),
                "blood_UA": round(blood_ua, 2),
            })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 自检：防止"数据太容易"被当成成绩
# --------------------------------------------------------------------------

def self_check(df: pd.DataFrame, warn_corr=0.55, fail_corr=0.70,
               warn_r2=0.85, fail_r2=0.95, seed=42):
    """
    生成后自检。返回 ``(ok, messages)``。

    阈值分两档，只朝「太容易」一个方向卡：

    - **警告档**（corr > 0.55 / R² > 0.85）：数据偏容易，需要人工确认生成器
      里没有混进确定性反演，但不阻断流程。
    - **失败档**（corr > 0.70 / R² > 0.95）：几乎可以断定存在确定性映射，
      这些数字一旦进报告就是「用合成数据冒充性能」，直接返回 False。

    分两档的理由：单档阈值在实际运行中会误伤——本生成器一次合法抽样就落在
    0.516，距 0.55 只差 0.034，换个种子极易被同一档阈值判死，而失败原因
    其实只是随机波动。真正需要拦的是「接近 1 的确定性关系」。

    注意：本函数只负责挡「太容易」。**「太难」不报错**——相关性偏低的合成
    数据只是保守，不构成夸大成果的风险。
    """
    msgs = []
    ok = True

    corr = float(np.corrcoef(df["sweat_UA"], df["blood_UA"])[0, 1])
    msgs.append(f"corr(sweat_UA, blood_UA) = {corr:.3f}"
                f"（期望 0.15–{warn_corr}，失败档 > {fail_corr}）")
    if abs(corr) > fail_corr:
        ok = False
        msgs.append(f"  !! 相关系数 {corr:.3f} > {fail_corr}：疑似确定性反演，"
                    "数据不得作为任何性能证据")
    elif abs(corr) > warn_corr:
        msgs.append(f"  !  相关系数 {corr:.3f} 略高于 {warn_corr}："
                    "请确认生成器无确定性映射（本次不阻断流程）")

    # 用协变量+标记物的 Ridge 快速五折，确认整体难度合理
    try:
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import GroupKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score

        feats = ["sweat_UA", "sweat_rate", "sweat_pH", "BMI", "age", "eGFR"]
        d = df.copy()
        # sex 在本生成器输出里是 'M'/'F'；显式映射而非依赖下游编码器，
        # 让自检脚本可以在没装 sklearn 的机器上也只跳过这一步。
        d["sex_enc"] = d["sex"].map({"M": 1, "F": 0})
        feats = feats + ["sex_enc"]
        d = d[feats + ["blood_UA", "subject_id"]].dropna()

        X = d[feats].to_numpy(float)
        y = d["blood_UA"].to_numpy(float)
        g = d["subject_id"].to_numpy()

        preds = np.full(len(d), np.nan)
        for tr, te in GroupKFold(n_splits=5).split(X, y, groups=g):
            pipe = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
            pipe.fit(X[tr], y[tr])
            preds[te] = pipe.predict(X[te])
        r2 = float(r2_score(y, preds))
        msgs.append(f"Ridge 五折分组 CV 的 R² = {r2:.3f}"
                    f"（期望 < {warn_r2}，失败档 > {fail_r2}）")
        if r2 > fail_r2:
            ok = False
            msgs.append(f"  !! R² {r2:.3f} > {fail_r2}：合成数据过于容易，"
                        "不得作为性能证据")
        elif r2 > warn_r2:
            msgs.append(f"  !  R² {r2:.3f} 略高于 {warn_r2}：请复核特征集，"
                        "本次不阻断流程")
    except ImportError:
        msgs.append("（未安装 scikit-learn，跳过 Ridge 快速自检）")

    rate = float(df["sweat_UA_censored"].mean() * 100)
    msgs.append(f"低于试剂盒 LOD({KIT_LOD_UMOL} μmol/L) 的左删失比例 = {rate:.1f}%")

    return ok, msgs


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    setup_console()
    parser = argparse.ArgumentParser(description="生成尿酸合成数据（仅流程验证）")
    parser.add_argument("--out", default="data/ua_example_merged_data.csv",
                        help="输出 CSV 路径")
    parser.add_argument("--n-subjects", type=int, default=50,
                        help="受试者数（默认 50，与验证方案目标样本量一致）")
    parser.add_argument("--n-timepoints", type=int, default=3,
                        help="每人采样点数（默认 3：T0/T1/T2）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-check", action="store_true",
                        help="跳过生成后自检")
    args = parser.parse_args()

    df = generate_synthetic_data(
        n_subjects=args.n_subjects,
        n_timepoints=args.n_timepoints,
        random_seed=args.seed,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig", lineterminator="\n")

    # 写合成数据标记：下游与报告据此强制打水印
    meta = {
        "synthetic": True,
        "generator": "src/ua_make_example_data.py",
        "seed": int(args.seed),
        "n_subjects": int(args.n_subjects),
        "n_timepoints": int(args.n_timepoints),
        "n_rows": int(len(df)),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kit_lod_umol": KIT_LOD_UMOL,
        "blood_unit": "umol/L",
        "caveat": CAVEAT,
    }
    meta_path = out.with_suffix("").with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8", newline="\n")

    print(f"[ua_make_example_data] 生成 {len(df)} 行"
          f"（{args.n_subjects} 受试者 × {args.n_timepoints} 时点）")
    print(f"[ua_make_example_data] 已写入 {out}")
    print(f"[ua_make_example_data] 合成数据标记 → {meta_path.name}")
    print(f"[ua_make_example_data] blood_UA 单位 μmol/L，范围 "
          f"{df['blood_UA'].min():.1f} – {df['blood_UA'].max():.1f} μmol/L"
          f"（男 >420 / 女 >360 视为高尿酸血症）")
    male_hu = ((df["sex"] == "M") & (df["blood_UA"] > 420)).mean() * 100
    female_hu = ((df["sex"] == "F") & (df["blood_UA"] > 360)).mean() * 100
    print(f"[ua_make_example_data] 合成高尿酸血症比例：男 {male_hu:.0f}% / 女 {female_hu:.0f}%"
          "（这是生成参数的结果，不是流行病学估计）")
    print(f"[ua_make_example_data] sweat_UA 范围 "
          f"{df['sweat_UA'].min():.2f} – {df['sweat_UA'].max():.2f} μmol/L")

    if not args.no_check:
        print("\n[ua_make_example_data] --- 生成后自检 ---")
        ok, msgs = self_check(df, seed=args.seed)
        for m in msgs:
            print("  " + m)
        if not ok:
            print("\n[ua_make_example_data] 自检未通过：合成数据过于容易，"
                  "不能用作性能证据。请调大噪声或移除确定性反演公式。")
            raise SystemExit(2)
        print("  → 自检通过：合成数据难度合理（仅可作流程验证）")


if __name__ == "__main__":
    main()
