#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ua_reproduce_figure5hi.py — 模型对比与结果可视化（Figure 5h-i 风格）
=====================================================================

来源与改动
----------
本文件改编自论文官方公开代码仓库 ``SIJIEJI/polycore-lipid-causal`` 的 ``src/reproduce_figure5hi.py``
（检测场景由脂质迁移为尿酸，结构与参数均有结构性改动，非照搬）。
详见 README「来源与署名」与 docs/UA_MIGRATION_GUIDE.md。


产出六类证据：

1. **主结果**：7 个模型 × 受试者分组 5 折交叉验证（MAE / R²），
   叠加个体化响应曲线 —— ``figure5hi.png`` + 逐折明细 + 折外预测。
2. **不确定度**：受试者簇 bootstrap（n=1000）置信区间 + 模型间配对差值 CI。
3. **2×2 消融**：{Ridge, RandomForest} × {全特征, 因果最小集}——
   把「因果特征集带来的增益」与「换个更强估计器带来的增益」分开。
   论文把这两者混在一起，因此它的"因果增益"实际无法归因；
   这是本项目相对论文的一处实质方法学改进。
4. **折划分敏感性**：随机 KFold vs 受试者分组 GroupKFold，
   量化「同一人跨折出现」带来的乐观偏差。
5. **肾功能分层**：按 eGFR 分两层（对齐验证方案排除 CKD stage 3+ 的入组标准）
   独立评估，界定模型适用边界。
6. **个体响应曲线**：固定个体协变量、扫描汗液尿酸输入，可视化个体化斜率。

原版存在的 8 处缺陷（本版逐一修掉）
----------------------------------
1. 特征列表在两处硬编码且互相矛盾（脚本里 Causal_ML 7 列 / spec 里 10 列）；
2. ``Causal_ML`` 与 ``Ridge`` 特征完全相同（都是 10 列），"因果增益"无法分离；
3. ``StandardScaler.fit_transform`` 在 CV 循环**之外**执行 → 数据泄漏；
4. 抖动 ``np.random.normal`` 未设种子 → 图不可复现；
5. 负对照参考线 ``.values[0]`` 在缺模型时 ``IndexError``；
6. 未设中文字体 → 中文与负号变方框；
7. 肾功能分层用测量条数判断样本是否充足（``len(subdf) < 10``），
   且上界 150 会把 eGFR>150 的超滤过者判成缺失；
8. ``evaluate_model_cv`` 收了 ``groups`` 形参却从未使用。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import ua_causal_specification as spec
from ua_data_utils import (
    EXCLUDED_RENAL_LABEL, STUDY_RENAL_LABELS,
    assign_renal_group, cluster_bootstrap_ci, dedup_columns, load_merged_data,
    make_group_splitter, oof_predict, paired_delta_ci, resolve_features,
    setup_console, summarize_oof, write_csv,
)

# 中文字体（否则中文与负号在图上会变成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

MODEL_COLORS = {
    "Simple": "#E8D5B7",
    "Multi": "#F4A261",
    "Ridge": "#2A9D8F",
    "Lasso": "#264653",
    "Causal_ML": "#E76F51",
    "Negative_Control": "#A8A8A8",
    "Negative_Control_Dynamic": "#CFCFCF",
}
MODEL_ORDER = list(MODEL_COLORS)

WATERMARK = "合成数据 · 仅流程验证 · 非性能证据"


def _add_watermark(fig, text=WATERMARK, enabled=True):
    """合成数据强制水印。写进图里，避免数字被截图后脱离上下文引用。"""
    if not enabled:
        return
    fig.text(0.5, 0.5, text, fontsize=22, color="#E76F51", alpha=0.13,
             ha="center", va="center", rotation=18, zorder=10,
             fontweight="bold")


def _unit_label(df, blood_unit_hint="umol/L"):
    return f"血尿酸（{blood_unit_hint}）"


# --------------------------------------------------------------------------
# 主结果
# --------------------------------------------------------------------------

def build_model_plan(marker_mode="gold"):
    """从规格模块构造 (模型名 → 特征列 / 估计器对象) 的计划表。"""
    feature_sets = spec.get_feature_sets()
    model_defs = spec.get_model_definitions()
    plan = {}
    for name, cfg in model_defs.items():
        key = cfg["feature_set_key"]
        cols = resolve_features(feature_sets[key]["columns"], marker_mode)
        plan[name] = {
            "features": cols,
            "estimator": spec.get_estimator(cfg["estimator"]),
            "estimator_key": cfg["estimator"],
            "feature_set_key": key,
            "tier": feature_sets[key]["tier"],
            "description": cfg["description"],
        }
    return plan


def evaluate_all_models(df, plan, n_splits=5, n_boot=1000, seed=42):
    """逐模型做受试者分组 CV，返回 (summary_df, fold_df, oof_by_model)。"""
    n_sub = int(df["subject_id"].nunique())

    summary_rows, fold_rows, oof_by_model = [], [], {}
    for name in MODEL_ORDER:
        if name not in plan:
            continue
        cfg = plan[name]
        feats = cfg["features"]
        missing = [c for c in feats if c not in df.columns]
        if missing:
            print(f"  {name:26s} | 跳过（缺列 {missing}）")
            continue

        oof, fold_info = oof_predict(
            df, feats, cfg["estimator"], make_group_splitter(n_sub, n_splits))
        oof_by_model[name] = oof

        overall = summarize_oof(oof, name)
        boot = {m: cluster_bootstrap_ci(oof, metric=m, n_boot=n_boot, seed=seed)
                for m in ("mae", "r2")}

        row = {
            "model": name,
            "feature_set_key": cfg["feature_set_key"],
            "estimator": cfg["estimator_key"],
            "tier": cfg["tier"],
            "n_features": len(feats),
            "features": ";".join(feats),
            "n_used": overall["n_used"],
            "n_subjects_used": overall["n_subjects_used"],
            "mae_mean": overall["mae"],
            "r2_mean": overall["r2"],
            "rmse_mean": overall["rmse"],
            "mae_ci_low": boot["mae"]["ci_low"],
            "mae_ci_high": boot["mae"]["ci_high"],
            "r2_ci_low": boot["r2"]["ci_low"],
            "r2_ci_high": boot["r2"]["ci_high"],
        }

        # 逐折指标
        fold_maes, fold_r2s = [], []
        for fi in fold_info:
            f = fi["fold"]
            sub = oof[oof["fold"] == f]
            s = summarize_oof(sub, f"fold{f}")
            fold_maes.append(s["mae"])
            fold_r2s.append(s["r2"])
            fold_rows.append({
                "model": name,
                "fold": f,
                "n_train": fi["n_train"],
                "n_test": fi["n_test"],
                "n_train_subjects": fi["n_train_subjects"],
                "n_test_subjects": fi["n_test_subjects"],
                "test_subject_ids": ";".join(str(v) for v in fi["test_subject_ids"]),
                "subject_overlap": ";".join(str(v) for v in fi["subject_overlap"]),
                "mae": s["mae"],
                "r2": s["r2"],
            })
        row["mae_fold_std"] = float(np.std(fold_maes))
        row["r2_fold_std"] = float(np.std(fold_r2s))
        row["fold_maes"] = fold_maes
        row["fold_r2s"] = fold_r2s

        summary_rows.append(row)
        print(f"  {name:26s} | {len(feats):2d} 列 | "
              f"MAE = {overall['mae']:6.2f} [{boot['mae']['ci_low']:.2f}, {boot['mae']['ci_high']:.2f}] | "
              f"R² = {overall['r2']:+.3f} [{boot['r2']['ci_low']:+.3f}, {boot['r2']['ci_high']:+.3f}]")

    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary["model"] = pd.Categorical(summary["model"], MODEL_ORDER, ordered=True)
        summary = summary.sort_values("model").reset_index(drop=True)
    return summary, pd.DataFrame(fold_rows), oof_by_model


def run_bootstrap(oof_by_model, n_boot=1000, seed=42):
    """受试者簇 bootstrap CI（逐模型逐指标）+ 模型间配对差值 CI。"""
    rows = []
    for name, oof in oof_by_model.items():
        for metric in ("mae", "r2"):
            res = cluster_bootstrap_ci(oof, metric=metric, n_boot=n_boot, seed=seed)
            rows.append({"model": name, **res})
    boot_df = pd.DataFrame(rows)

    # 配对比较：各模型 vs 两个负对照，以及 Causal_ML vs Ridge（关键对比）
    pairs = []
    ref_list = [m for m in ("Negative_Control", "Negative_Control_Dynamic",
                            "Simple", "Ridge") if m in oof_by_model]
    for name, oof in oof_by_model.items():
        for ref in ref_list:
            if name == ref:
                continue
            for metric in ("r2", "mae"):
                _, s = paired_delta_ci(oof, oof_by_model[ref], metric=metric,
                                       n_boot=n_boot, seed=seed)
                pairs.append({"model_A": name, "model_B": ref, **s})
    pair_df = pd.DataFrame(pairs)
    return boot_df, pair_df


# --------------------------------------------------------------------------
# 2×2 消融：把「特征集增益」与「估计器增益」分开
# --------------------------------------------------------------------------

def run_ablation(df, marker_mode="gold", n_splits=5, n_boot=1000, seed=42):
    feature_sets = spec.get_feature_sets()
    n_sub = int(df["subject_id"].nunique())

    cells = []
    for est_key in spec.ABLATION_ESTIMATORS:
        for fs_key in spec.ABLATION_FEATURE_SETS:
            feats = resolve_features(feature_sets[fs_key]["columns"], marker_mode)
            missing = [c for c in feats if c not in df.columns]
            if missing:
                continue
            oof, _ = oof_predict(df, feats, spec.get_estimator(est_key),
                                 make_group_splitter(n_sub, n_splits))
            s = summarize_oof(oof, f"{est_key}|{fs_key}")
            b = cluster_bootstrap_ci(oof, metric="r2", n_boot=n_boot, seed=seed)
            cells.append({
                "estimator": est_key,
                "feature_set_key": fs_key,
                "n_features": len(feats),
                "r2_mean": s["r2"], "r2_ci_low": b["ci_low"], "r2_ci_high": b["ci_high"],
                "mae_mean": s["mae"],
                "n_used": s["n_used"], "n_subjects_used": s["n_subjects_used"],
            })
    cell_df = pd.DataFrame(cells)
    if len(cell_df) == 0:
        return cell_df, pd.DataFrame()

    cell_df["estimator_short"] = cell_df["estimator"].str.split("(").str[0]

    # 边际效应：固定另一因子取平均，得到可归因的增益分解
    r2 = cell_df.pivot(index="estimator_short", columns="feature_set_key", values="r2_mean")
    rows = []
    for est in r2.index:
        rows.append({"effect": "feature_set_gain（全特征 → 因果最小集，固定估计器）",
                     "level": est,
                     "delta_r2": float(r2.loc[est].get("causal_guided_minimal", np.nan)
                                       - r2.loc[est].get("full_model", np.nan))})
    for fs in r2.columns:
        rows.append({"effect": "estimator_gain（Ridge → RandomForest，固定特征集）",
                     "level": fs,
                     "delta_r2": float(r2[fs].get("RandomForest", np.nan)
                                       - r2[fs].get("Ridge", np.nan))})
    marginal_df = pd.DataFrame(rows)
    marginal_df["mean_delta_r2"] = marginal_df.groupby("effect")["delta_r2"].transform("mean")

    return cell_df, marginal_df


def plot_ablation_heatmap(cell_df, out_path, synthetic=True):
    if len(cell_df) == 0:
        return
    est_order = [e.split("(")[0] for e in spec.ABLATION_ESTIMATORS]
    fs_order = list(spec.ABLATION_FEATURE_SETS)
    grid = np.full((len(est_order), len(fs_order)), np.nan)

    for _, r in cell_df.iterrows():
        if r["estimator_short"] in est_order and r["feature_set_key"] in fs_order:
            grid[est_order.index(r["estimator_short"]),
                 fs_order.index(r["feature_set_key"])] = r["r2_mean"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    im = ax.imshow(grid, cmap="YlOrRd", aspect="auto", vmin=np.nanmin(grid) - 0.02,
                   vmax=np.nanmax(grid) + 0.02)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isfinite(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color="#222222")
    ax.set_xticks(range(len(fs_order)))
    ax.set_xticklabels([f"{k}\n({spec.get_feature_sets()[k]['columns'].__len__()} 列)"
                        for k in fs_order], fontsize=9)
    ax.set_yticks(range(len(est_order)))
    ax.set_yticklabels(est_order, fontsize=10)
    ax.set_xlabel("特征集")
    ax.set_ylabel("估计器")
    ax.set_title("2×2 消融：特征集增益 vs 估计器增益（折外 R²）")
    fig.colorbar(im, ax=ax, label="R²", shrink=0.85)
    _add_watermark(fig, enabled=synthetic)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


# --------------------------------------------------------------------------
# 折划分敏感性：量化受试者泄漏带来的乐观偏差
# --------------------------------------------------------------------------

def run_sensitivity_analysis(df, marker_mode="gold", n_splits=5,
                             n_boot=1000, seed=42):
    """
    同一个模型、同一个特征集，只改折划分方式：
    随机 KFold（同一人会跨折） vs 受试者分组 GroupKFold（同一人不跨折）。
    差值就是「受试者泄漏」带来的乐观偏差。
    """
    from ua_data_utils import kfold_optimism_delta

    feature_sets = spec.get_feature_sets()
    rows = []
    for fs_key in ("causal_guided_minimal_renal", "full_model"):
        feats = resolve_features(feature_sets[fs_key]["columns"], marker_mode)
        if any(c not in df.columns for c in feats):
            continue
        d = kfold_optimism_delta(
            df, feats, spec.get_estimator("Ridge(alpha=1.0)"),
            n_splits=n_splits, seed=seed, metric="r2", n_boot=n_boot)
        rows.append({"feature_set_key": fs_key, "n_features": len(feats), **d})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 肾功能分层
# --------------------------------------------------------------------------

def evaluate_by_renal_function(df, marker_mode="gold", n_splits=5,
                               n_boot=1000, seed=42, min_subjects=15):
    """
    按 eGFR 分层独立评估因果模型。

    两层口径（``eGFR 60-89`` / ``eGFR>=90``）来自验证方案的入组标准：
    **排除 CKD stage 3+（eGFR<60）**，所以 <60 这一层在本研究里不可分析，
    这在报告中如实说明，而不是硬凑成三层。

    判据用**受试者数**而非测量条数：30 个人各测 6 次是 180 条，
    但独立信息量仍然只有 30。
    """
    feature_sets = spec.get_feature_sets()
    feats = resolve_features(
        feature_sets["causal_guided_minimal_renal"]["columns"], marker_mode)
    estimator = spec.get_estimator("Ridge(alpha=1.0)")

    df = df.copy()
    df["renal_group"] = assign_renal_group(df["eGFR"], tiers="study")

    excluded = df[df["renal_group"] == EXCLUDED_RENAL_LABEL]
    n_excluded_subjects = int(excluded["subject_id"].nunique())

    results = []
    for group in STUDY_RENAL_LABELS:
        sub = df[df["renal_group"] == group]
        n_sub = int(sub["subject_id"].nunique())
        n_meas = int(len(sub))

        if n_sub < min_subjects:
            results.append({
                "renal_group": group, "n_subjects": n_sub, "n_measurements": n_meas,
                "mae_mean": np.nan, "mae_ci_low": np.nan, "mae_ci_high": np.nan,
                "r2_mean": np.nan, "r2_ci_low": np.nan, "r2_ci_high": np.nan,
                "n_splits_used": 0,
                "skip_reason": f"该层受试者仅 {n_sub} 人（< {min_subjects}），"
                               f"分组交叉验证的估计不稳定，故如实跳过",
            })
            continue

        oof, _ = oof_predict(sub, feats, estimator,
                             make_group_splitter(n_sub, n_splits))
        s = summarize_oof(oof, group)
        bm = cluster_bootstrap_ci(oof, metric="mae", n_boot=n_boot, seed=seed)
        br = cluster_bootstrap_ci(oof, metric="r2", n_boot=n_boot, seed=seed)
        results.append({
            "renal_group": group, "n_subjects": n_sub, "n_measurements": n_meas,
            "mae_mean": s["mae"], "mae_ci_low": bm["ci_low"], "mae_ci_high": bm["ci_high"],
            "r2_mean": s["r2"], "r2_ci_low": br["ci_low"], "r2_ci_high": br["ci_high"],
            "n_splits_used": max(2, min(n_splits, n_sub)),
            "skip_reason": "",
        })

    res = pd.DataFrame(results)

    # 被排除层也落一行，而不是只打印一行字。
    # 理由：报告要能"逐行审计"这张表——读者看到只有两层时，
    # 必须能从产物本身知道第三层去哪了、为什么不在，以及到底排除了几个人。
    # 只在 stdout 说一句、CSV 里查无此事，等于让审计者只能相信作者的转述。
    res = pd.concat([res, pd.DataFrame([{
        "renal_group": EXCLUDED_RENAL_LABEL,
        "n_subjects": n_excluded_subjects,
        "n_measurements": int(len(excluded)),
        "mae_mean": np.nan, "mae_ci_low": np.nan, "mae_ci_high": np.nan,
        "r2_mean": np.nan, "r2_ci_low": np.nan, "r2_ci_high": np.nan,
        "n_splits_used": 0,
        "skip_reason": "验证方案入组标准排除 CKD stage 3+（eGFR<60），"
                       "该层不属于本研究目标人群，故不参与分层分析",
    }])], ignore_index=True)

    res.attrs["n_excluded_subjects_eGFR_lt60"] = n_excluded_subjects

    # 供绘图使用的一句话跳过说明：CSV 里的 skip_reason 是完整理由，
    # 图上放不下，这里另给一个短标签。
    res["skip_label"] = [
        ("方案排除\n不参与分层" if g == EXCLUDED_RENAL_LABEL
         else ("样本不足\n如实跳过" if s else ""))
        for g, s in zip(res["renal_group"], res["skip_reason"])
    ]
    return res


def plot_renal_stratified(res, out_path, synthetic=True):
    d = res.copy()
    if len(d) == 0:
        return
    x = np.arange(len(d))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ok = d["r2_mean"].notna()

    axes[0].bar(x[ok], d.loc[ok, "mae_mean"],
                yerr=[d.loc[ok, "mae_mean"] - d.loc[ok, "mae_ci_low"],
                      d.loc[ok, "mae_ci_high"] - d.loc[ok, "mae_mean"]],
                color="#E76F51", edgecolor="black", capsize=4)
    axes[1].bar(x[ok], d.loc[ok, "r2_mean"],
                yerr=[d.loc[ok, "r2_mean"] - d.loc[ok, "r2_ci_low"],
                      d.loc[ok, "r2_ci_high"] - d.loc[ok, "r2_mean"]],
                color="#2A9D8F", edgecolor="black", capsize=4)

    for ax, label in zip(axes, ("MAE（μmol/L）", "折外 R²")):
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}\n(n={int(n)} 人)" for g, n in
                            zip(d["renal_group"], d["n_subjects"])], fontsize=9)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, axis="y")
        for i, (is_ok, lab) in enumerate(zip(ok, d["skip_label"])):
            if not is_ok and lab:
                ax.text(i, 0.02, lab, ha="center", va="bottom",
                        fontsize=8.5, color="#888888", transform=ax.get_xaxis_transform())
    axes[1].axhline(0, color="gray", linewidth=0.8, alpha=0.5)
    fig.suptitle("因果模型在肾功能亚组内的表现（分层验证适用边界）", fontsize=11)
    _add_watermark(fig, enabled=synthetic)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


# --------------------------------------------------------------------------
# 个体响应曲线
# --------------------------------------------------------------------------

def plot_individual_response_curves(df, out_path, marker_mode="gold",
                                    n_points=60, synthetic=True):
    """
    固定个体协变量、扫描汗液尿酸输入，画出个体特异的「汗液 → 血液」映射斜率。
    肾功能不同的人斜率不同 —— 这正是需要个体化校准的依据。
    """
    feature_sets = spec.get_feature_sets()
    feats = resolve_features(
        feature_sets["causal_guided_minimal_renal"]["columns"], marker_mode)
    marker = feats[0]

    sub = df[dedup_columns(feats, ["blood_UA", "eGFR", "subject_id"])].dropna()
    if len(sub) < 10:
        print("[plot] 个体响应曲线：样本不足，跳过")
        return

    pipe = Pipeline([("scaler", StandardScaler()),
                     ("model", spec.get_estimator("Ridge(alpha=1.0)"))])
    pipe.fit(sub[feats].to_numpy(float), sub["blood_UA"].to_numpy(float))

    df2 = df.copy()
    df2["renal_group"] = assign_renal_group(df2["eGFR"], tiers="study")
    colors = {"eGFR 60-89": "#E76F51", "eGFR>=90": "#2A9D8F"}
    grid = np.linspace(np.percentile(sub[marker], 2),
                       np.percentile(sub[marker], 98), n_points)

    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    slopes = []
    for group in STUDY_RENAL_LABELS:
        gsub = df2[df2["renal_group"] == group]
        if gsub.empty:
            continue
        pick = gsub.iloc[len(gsub) // 2]
        base = pick[feats].to_numpy(dtype=float)
        preds = []
        for v in grid:
            base[0] = float(v)
            preds.append(float(pipe.predict(base.reshape(1, -1))[0]))
        slope, intercept = (float(x) for x in np.polyfit(grid, preds, 1))
        slopes.append(slope)
        ax.plot(grid, preds, color=colors.get(group, "#333333"), linewidth=2.4,
                label=f"{group}（代表受试者 eGFR={pick['eGFR']:.0f}）\n"
                      f"    斜率 {slope:.2f}，截距 {intercept:.0f} μmol/L")

    # 斜率相同、截距不同时，把这件事直接写在图上。
    # 否则读者看到"两条线斜率一样"会以为是绘图缺陷，或者误以为
    # 方案里的"个体化"没实现；其实是本合成数据生成器**没有**编码
    # 肾功能对汗/血转运比的调制，斜率相同属于预期，不是 bug。
    if len(slopes) == 2 and abs(slopes[0] - slopes[1]) < 0.02:
        ax.text(0.02, 0.96,
                "注：本图两条曲线斜率相同，是因为当前合成数据生成器未编码\n"
                "肾功能对汗/血转运比的调制（合成数据仅用于流程验证）。\n"
                "真实数据上斜率是否随肾功能分层而不同，属待检验假设。",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                color="#666666",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#F7F7F7",
                          edgecolor="#DDDDDD", alpha=0.9))

    ax.set_xlabel(f"汗液尿酸输入（{'μmol/L' if marker_mode == 'gold' else '电极响应（相对单位）'}）")
    ax.set_ylabel("模型预测的血尿酸（μmol/L）")
    ax.set_title("个体化响应曲线：固定该受试者全部协变量，仅扫描汗液尿酸")
    ax.legend(title="肾功能分层", fontsize=8.5)
    ax.grid(alpha=0.3)
    _add_watermark(fig, enabled=synthetic)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


# --------------------------------------------------------------------------
# 主图（Figure 5h-i 风格）
# --------------------------------------------------------------------------

def plot_figure5hi(summary, out_path, seed=42, synthetic=True):
    if len(summary) == 0:
        return
    d = summary.reset_index(drop=True)
    x = np.arange(len(d))
    colors = [MODEL_COLORS.get(m, "#333333") for m in d["model"]]
    rng = np.random.default_rng(seed)          # 固定种子 → 图可复现

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2))

    # ---- 左：MAE
    ax = axes[0]
    ax.bar(x, d["mae_mean"],
           yerr=[d["mae_mean"] - d["mae_ci_low"], d["mae_ci_high"] - d["mae_mean"]],
           color=colors, edgecolor="black", linewidth=0.8, capsize=4,
           error_kw={"elinewidth": 1.2})
    for i, row in d.iterrows():
        jit = rng.uniform(-0.045, 0.045, size=len(row["fold_maes"]))
        ax.scatter(i + jit, row["fold_maes"], color="black", s=13,
                   zorder=3, alpha=0.55)
    nc = d[d["model"] == "Negative_Control"]["mae_mean"]
    if len(nc):
        ax.axhline(float(nc.iloc[0]), color="gray", linestyle="--", alpha=0.65,
                   label=f"静态负对照 = {float(nc.iloc[0]):.1f}")
        ax.legend(fontsize=8)
    ax.set_ylabel("MAE（μmol/L）")
    ax.set_title("血尿酸预测误差（越小越好）")
    ax.grid(alpha=0.25, axis="y")

    # ---- 右：R²
    ax = axes[1]
    ax.bar(x, d["r2_mean"],
           yerr=[d["r2_mean"] - d["r2_ci_low"], d["r2_ci_high"] - d["r2_mean"]],
           color=colors, edgecolor="black", linewidth=0.8, capsize=4,
           error_kw={"elinewidth": 1.2})
    for i, row in d.iterrows():
        jit = rng.uniform(-0.045, 0.045, size=len(row["fold_r2s"]))
        ax.scatter(i + jit, row["fold_r2s"], color="black", s=13,
                   zorder=3, alpha=0.55)
    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.45)
    ax.set_ylabel("折外 R²")
    ax.set_title("血尿酸预测决定系数（越大越好）")
    ax.grid(alpha=0.25, axis="y")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m}\n({int(n)} 列)" for m, n in
                            zip(d["model"], d["n_features"])], rotation=25,
                           ha="right", fontsize=8.5)

    fig.suptitle("汗液尿酸 → 血尿酸映射：7 模型受试者分组 5 折交叉验证"
                 "（误差棒 = 受试者簇 bootstrap 95% CI；黑点 = 各折值）",
                 fontsize=10.5)
    _add_watermark(fig, enabled=synthetic)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    setup_console()
    parser = argparse.ArgumentParser(description="UA Figure 5h-i 复现与扩展")
    parser.add_argument("--data", default="data/ua_example_merged_data.csv")
    parser.add_argument("--out", default="results/ua_figure5hi")
    parser.add_argument("--blood-unit", default="umol/L",
                        choices=["umol/L", "mg/dL", "auto"])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--marker-mode", default="gold",
                        choices=["gold", "sensor"])
    parser.add_argument("--min-stratum-subjects", type=int, default=15,
                        help="肾功能分层分析要求的最少受试者数，不足则如实跳过")
    parser.add_argument("--no-ablation", action="store_true")
    parser.add_argument("--no-sensitivity", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df, meta = load_merged_data(args.data, blood_unit=args.blood_unit)
    synth = bool(meta.get("synthetic_flag"))
    if synth:
        print(f"[ua_reproduce_figure5hi] 注意：{meta.get('caveat')}")
    print(f"[ua_reproduce_figure5hi] 载入 {meta['n_rows']} 行 / "
          f"{meta['n_subjects']} 名受试者；标记物模式 = {args.marker_mode}")
    print(f"[ua_reproduce_figure5hi] 血尿酸范围 "
          f"{df['blood_UA'].min():.1f} – {df['blood_UA'].max():.1f} μmol/L")

    plan = build_model_plan(args.marker_mode)

    # ---- 1. 主结果
    print("\n--- 受试者分组交叉验证（主结果） ---")
    summary, folds, oof_by_model = evaluate_all_models(
        df, plan, n_splits=args.n_splits, n_boot=args.n_boot, seed=args.seed)

    write_csv(summary.drop(columns=["fold_maes", "fold_r2s"]),
              out_dir / "figure5hi_summary_metrics.csv")
    write_csv(folds, out_dir / "figure5hi_fold_metrics.csv")

    oof_all = pd.concat(
        [o.assign(model=name) for name, o in oof_by_model.items()],
        ignore_index=True)[["model", "subject_id", "time_point", "y_true",
                            "y_pred", "fold"]]
    write_csv(oof_all, out_dir / "figure5hi_oof_predictions.csv")

    # ---- 2. Bootstrap 与配对比较
    print("\n--- 受试者簇 bootstrap（n = %d）与配对比较 ---" % args.n_boot)
    boot_df, pair_df = run_bootstrap(oof_by_model, n_boot=args.n_boot, seed=args.seed)
    write_csv(boot_df, out_dir / "bootstrap_ci.csv")
    write_csv(pair_df, out_dir / "paired_delta_ci.csv")
    key = pair_df[(pair_df["model_A"] == "Causal_ML") & (pair_df["metric"] == "r2")]
    for _, r in key.iterrows():
        print(f"  Causal_ML vs {r['model_B']:26s} | ΔR² = {r['point_delta']:+.3f} "
              f"[{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] | "
              f"CI 不含 0 = {r['ci_excludes_zero']}")

    # ---- 3. 2×2 消融
    if not args.no_ablation:
        print("\n--- 2×2 消融：特征集增益 vs 估计器增益 ---")
        cell_df, marginal_df = run_ablation(
            df, args.marker_mode, args.n_splits, args.n_boot, args.seed)
        write_csv(cell_df, out_dir / "ablation_estimator_vs_featureset.csv")
        if len(marginal_df):
            write_csv(marginal_df, out_dir / "ablation_marginal_effects.csv")
            for _, r in marginal_df.drop_duplicates(
                    subset=["effect"]).iterrows():
                print(f"  {r['effect']:52s} | 平均 ΔR² = {r['mean_delta_r2']:+.3f}")
        plot_ablation_heatmap(cell_df, out_dir / "ablation_heatmap.png", synthetic=synth)

    # ---- 4. 折划分敏感性
    if not args.no_sensitivity:
        print("\n--- 折划分敏感性：随机 KFold vs 受试者分组 GroupKFold ---")
        sens = run_sensitivity_analysis(
            df, args.marker_mode, args.n_splits, args.n_boot, args.seed)
        write_csv(sens, out_dir / "sensitivity_splitter.csv")
        for _, r in sens.iterrows():
            print(f"  {r['feature_set_key']:26s} | 分组 = {r['group_point']:+.3f} | "
                  f"随机 = {r['shuffle_point']:+.3f} | "
                  f"乐观偏差 Δ = {r['delta_shuffle_vs_group']:+.3f} "
                  f"[{r['delta_ci_low']:+.3f}, {r['delta_ci_high']:+.3f}]")

    # ---- 5. 肾功能分层
    print("\n--- 肾功能分层（两层口径，对齐入组排除标准）---")
    renal = evaluate_by_renal_function(
        df, args.marker_mode, args.n_splits, args.n_boot, args.seed,
        min_subjects=args.min_stratum_subjects)
    renal["n_excluded_subjects_eGFR_lt60"] = renal.attrs.get(
        "n_excluded_subjects_eGFR_lt60", 0)
    write_csv(renal, out_dir / "renal_stratified_metrics.csv")
    for _, r in renal.iterrows():
        if r["skip_reason"]:
            print(f"  {r['renal_group']:14s} | n = {int(r['n_subjects'])} 人 | 跳过：{r['skip_reason']}")
        else:
            print(f"  {r['renal_group']:14s} | n = {int(r['n_subjects'])} 人 | "
                  f"MAE = {r['mae_mean']:.2f} | R² = {r['r2_mean']:+.3f}")
    if renal.attrs.get("n_excluded_subjects_eGFR_lt60"):
        print(f"  （另有 {renal.attrs['n_excluded_subjects_eGFR_lt60']} 名受试者的 "
              f"eGFR<60，按验证方案入组标准本就不应入组，故不参与分层）")

    # ---- 6. 出图
    print("\n--- 绘图 ---")
    plot_figure5hi(summary, out_dir / "figure5hi.png", seed=args.seed, synthetic=synth)
    plot_individual_response_curves(
        df, out_dir / "individual_response_curves.png",
        marker_mode=args.marker_mode, synthetic=synth)
    plot_renal_stratified(renal, out_dir / "renal_stratified.png", synthetic=synth)

    # ---- 结论
    print("\n--- 结果解读 ---")
    def _row(m):
        sel = summary.loc[summary["model"] == m]
        return sel.iloc[0] if len(sel) else None

    s, n, c = _row("Simple"), _row("Negative_Control"), _row("Causal_ML")
    for label, r in (("单标记基线 Simple", s),
                     ("静态负对照 NegControl", n),
                     ("因果模型 Causal_ML", c)):
        if r is not None:
            print(f"  {label:22s} | 特征 {int(r['n_features'])} 列 | "
                  f"R² = {r['r2_mean']:+.3f} | MAE = {r['mae_mean']:.2f} μmol/L")
    if c is not None and n is not None:
        if c["r2_mean"] > n["r2_mean"]:
            print("  → 含汗液标记物的因果模型优于纯协变量负对照。"
                  "（这是合成数据上的流程验证结果，不构成性能证据）")
        else:
            print("  → 因果模型未优于负对照：说明当前生成器赋予汗液标记物的"
                  "信息量不足。应调整生成假设后重跑，"
                  "而不是把这个数字写进报告当结论。")

    print(f"\n[ua_reproduce_figure5hi] 全部产物写入 {out_dir}")


if __name__ == "__main__":
    main()
