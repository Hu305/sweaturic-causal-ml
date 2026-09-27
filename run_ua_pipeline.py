#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_ua_pipeline.py — 尿酸因果 ML 管线一键运行入口
===================================================


四个步骤串行执行：

1. ``src/ua_make_example_data.py``    生成合成数据（仅流程验证）
2. ``src/ua_causal_specification.py`` 写出 DAG / 调整集推导 / 特征集 / 模型定义
3. ``src/ua_run_causal_adjustment.py`` 调整阶梯诊断（含 adjustment_ladder.png）
4. ``src/ua_reproduce_figure5hi.py``  7 模型分组 CV + Bootstrap + 消融 + 敏感性 + 分层

用法::

    # 全流程（合成数据 smoke test）
    python run_ua_pipeline.py

    # 真实数据：替换 data/merged_data.csv 后
    python run_ua_pipeline.py --data data/merged_data.csv --skip-smoke

    # 只重跑第 3、4 步，Bootstrap 次数降到 200 快速看结果
    python run_ua_pipeline.py --steps 3,4 --n-boot 200

    # 额外写出带 sha256 校验值的产物清单
    python run_ua_pipeline.py --manifest

设计要点
--------------
- ``subprocess.run([sys.executable, ...])`` 直接以参数列表调用，
  **不走 shell**（早期草稿用 ``shell=True`` 拼字符串，路径带空格或中文就可能出错，
  也是命令注入的常见来源）；
- 所有路径相对 ``REPO_ROOT`` 解析，脚本在任何工作目录下调用都成立；
- 新增 ``--steps`` / ``--seed`` / ``--marker-mode`` / ``--manifest``；
- 新增运行后产物校验：文件存在、非空、行数达标，并做三项机检
  （特征集是否真的不同、折划分有无受试者泄漏、Bootstrap 行数是否齐全）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
SRC = REPO_ROOT / "src"

sys.path.insert(0, str(SRC))
from ua_data_utils import setup_console  # noqa: E402

STEPS = {
    1: "生成合成数据（ua_make_example_data）",
    2: "写出因果规格（ua_causal_specification）",
    3: "调整阶梯诊断（ua_run_causal_adjustment）",
    4: "模型对比与扩展分析（ua_reproduce_figure5hi）",
}


def run_script(script: str, args: list[str]) -> None:
    """以参数列表方式调用子脚本（不使用 shell）。"""
    cmd = [sys.executable, str(SRC / script), *[str(a) for a in args]]
    print(f"\n{'=' * 68}\n>>> {' '.join(cmd)}\n{'=' * 68}", flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# 产物校验
# --------------------------------------------------------------------------

def _check_csv(path: Path, min_rows: int = 1, label: str = "") -> dict:
    ok = path.exists() and path.stat().st_size > 0
    n_rows, n_cols = 0, 0
    if ok:
        try:
            d = pd.read_csv(path, encoding="utf-8-sig")
            n_rows, n_cols = d.shape
            if n_rows < min_rows:
                ok = False
        except Exception:  # noqa: BLE001
            ok = False
    return {"file": path.name, "kind": "csv", "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "rows": n_rows, "cols": n_cols, "min_rows": min_rows,
            "ok": bool(ok), "label": label}


def _check_file(path: Path, label: str = "", min_bytes: int = 1024) -> dict:
    ok = path.exists() and path.stat().st_size >= min_bytes
    return {"file": path.name, "kind": path.suffix.lstrip("."),
            "exists": path.exists(),
            "bytes": path.stat().st_size if path.exists() else 0,
            "rows": None, "cols": None, "min_rows": None,
            "ok": bool(ok), "label": label}


def verify_outputs(data_path: Path, spec_dir: Path, adj_dir: Path,
                   fig_dir: Path, n_boot: int, steps: list[int]) -> tuple[bool, list[dict]]:
    checks: list[dict] = []
    if 1 in steps:
        checks.append(_check_csv(data_path, 30, "合成数据"))
    if 2 in steps:
        checks += [
            _check_csv(spec_dir / "causal_dag_edges.csv", 22, "DAG 边"),
            _check_csv(spec_dir / "dag_validation.csv", 1, "DAG 无环性校验"),
            _check_csv(spec_dir / "adjustment_set_derivation.csv", 5, "调整集推导"),
            _check_csv(spec_dir / "causal_feature_sets.csv", 7, "特征集"),
            _check_csv(spec_dir / "model_definitions.csv", 7, "模型定义"),
        ]
    if 3 in steps:
        checks += [
            _check_csv(adj_dir / "causal_adjustment_summary.csv", 5, "调整阶梯"),
            _check_file(adj_dir / "adjustment_ladder.png", "调整阶梯图"),
        ]
    if 4 in steps:
        checks += [
            _check_csv(fig_dir / "figure5hi_summary_metrics.csv", 6, "主结果"),
            _check_csv(fig_dir / "figure5hi_fold_metrics.csv", 25, "逐折明细"),
            _check_csv(fig_dir / "figure5hi_oof_predictions.csv", 30, "折外预测"),
            _check_csv(fig_dir / "bootstrap_ci.csv", 12, "Bootstrap CI"),
            _check_csv(fig_dir / "paired_delta_ci.csv", 5, "配对差值 CI"),
            _check_csv(fig_dir / "ablation_estimator_vs_featureset.csv", 4, "2×2 消融"),
            _check_csv(fig_dir / "sensitivity_splitter.csv", 2, "折划分敏感性"),
            _check_csv(fig_dir / "renal_stratified_metrics.csv", 2, "肾功能分层"),
            _check_file(fig_dir / "figure5hi.png", "主图"),
            _check_file(fig_dir / "individual_response_curves.png", "个体响应曲线"),
            _check_file(fig_dir / "renal_stratified.png", "肾功能分层图"),
            _check_file(fig_dir / "ablation_heatmap.png", "消融热图"),
        ]

    # ---- 机检 ①：Bootstrap 行数 = 模型数 × 2 指标，且 n_boot 一致
    bc = fig_dir / "bootstrap_ci.csv"
    if 4 in steps and bc.exists():
        d = pd.read_csv(bc, encoding="utf-8-sig")
        n_models = d["model"].nunique()
        expect = n_models * 2
        ok = (len(d) == expect) and (int(d["n_boot"].min()) == n_boot)
        checks.append({"file": "bootstrap_ci.csv[行数=模型数×2]", "kind": "assert",
                       "exists": True, "bytes": 0, "rows": len(d), "cols": None,
                       "min_rows": expect, "ok": bool(ok),
                       "label": f"模型 {n_models} 个 × 2 指标 = {expect} 行，n_boot={n_boot}"})

    # ---- 机检 ②：Causal_ML 与 Ridge 的特征集必须不同（否则"因果增益"无法分离）
    sm = fig_dir / "figure5hi_summary_metrics.csv"
    if 4 in steps and sm.exists():
        d = pd.read_csv(sm, encoding="utf-8-sig")
        def _nf(m):
            sel = d.loc[d["model"] == m, "n_features"]
            return int(sel.iloc[0]) if len(sel) else None
        c, r = _nf("Causal_ML"), _nf("Ridge")
        ok = (c is not None and r is not None and c != r)
        checks.append({"file": "figure5hi_summary_metrics.csv[特征集互异]", "kind": "assert",
                       "exists": True, "bytes": 0, "rows": None, "cols": None,
                       "min_rows": None, "ok": bool(ok),
                       "label": f"Causal_ML {c} 列 vs Ridge {r} 列（必须不同）"})

    # ---- 机检 ③：折划分无受试者泄漏
    fm = fig_dir / "figure5hi_fold_metrics.csv"
    if 4 in steps and fm.exists():
        d = pd.read_csv(fm, encoding="utf-8-sig")
        overlap = d["subject_overlap"].fillna("").astype(str).str.strip()
        n_bad = int((overlap != "").sum())
        checks.append({"file": "figure5hi_fold_metrics.csv[无泄漏]", "kind": "assert",
                       "exists": True, "bytes": 0, "rows": len(d), "cols": None,
                       "min_rows": None, "ok": n_bad == 0,
                       "label": f"训练折与测试折受试者无交集（违规折数 = {n_bad}）"})

    # ---- 机检 ④：调整阶梯的 r 必须是有限数
    ac = adj_dir / "causal_adjustment_summary.csv"
    if 3 in steps and ac.exists():
        d = pd.read_csv(ac, encoding="utf-8-sig")
        n_ok = int(d["r"].notna().sum())
        ok = n_ok >= 5 and int(d.loc[d["r"].notna(), "n_subjects"].min()) > 0
        checks.append({"file": "causal_adjustment_summary.csv[r 有限]", "kind": "assert",
                       "exists": True, "bytes": 0, "rows": len(d), "cols": None,
                       "min_rows": None, "ok": bool(ok),
                       "label": f"{n_ok} 个调整台阶梯产出有限 r 且 n_subjects>0"})

    all_ok = all(c["ok"] for c in checks)
    return all_ok, checks


def print_table(checks: list[dict]) -> None:
    print(f"\n{'产物文件':44s} {'类型':6s} {'字节':>9s} {'行数':>6s}  状态")
    print("-" * 78)
    for c in checks:
        rows = "" if c["rows"] is None else str(c["rows"])
        status = "OK" if c["ok"] else "FAIL"
        print(f"{c['file']:44s} {c['kind']:6s} {c['bytes']:9d} {rows:>6s}  {status}"
              + (f"   {c['label']}" if not c["ok"] and c["label"] else ""))
    print("-" * 78)
    n_bad = sum(1 for c in checks if not c["ok"])
    print(f"合计 {len(checks)} 项，通过 {len(checks) - n_bad} 项，失败 {n_bad} 项")


def write_manifest(paths: list[Path], out_path: Path) -> None:
    entries = []
    for p in sorted(set(paths)):
        if p.exists() and p.is_file():
            entries.append({
                "path": str(p.relative_to(REPO_ROOT)).replace("\\", "/"),
                "bytes": p.stat().st_size,
                "sha256": sha256_of(p),
            })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "generated_by": "run_ua_pipeline.py",
        "note": "sha256 用于校验报告附录引用的产物未被改动",
        "n_files": len(entries),
        "files": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")
    print(f"\n[manifest] 已写入 {out_path}（{len(entries)} 个文件）")


# --------------------------------------------------------------------------

def main():
    setup_console()
    parser = argparse.ArgumentParser(
        description="UA 因果 ML 管线一键运行",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/ua_example_merged_data.csv",
                        help="配对数据 CSV 路径（相对仓库根目录）")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="跳过第 1 步（使用真实数据时必须加）")
    parser.add_argument("--steps", default="all",
                        help="要执行的步骤，逗号分隔，如 '3,4'；默认 all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=1000,
                        help="bootstrap 重采样次数（报告口径为 1000）")
    parser.add_argument("--marker-mode", default="gold",
                        choices=["gold", "sensor"],
                        help="gold=酶法 sweat_UA；sensor=电极原始响应 sweat_UA_sensor")
    parser.add_argument("--manifest", action="store_true",
                        help="额外写出 results/MANIFEST.json（含 sha256）")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过运行后的产物校验")
    args = parser.parse_args()

    if args.steps.strip().lower() == "all":
        steps = sorted(STEPS)
    else:
        try:
            steps = sorted({int(s) for s in args.steps.split(",") if s.strip()})
        except ValueError:
            parser.error(f"--steps 需要形如 '3,4' 的整数列表，收到 {args.steps!r}")
        bad = [s for s in steps if s not in STEPS]
        if bad:
            parser.error(f"未知步骤 {bad}，可用 {sorted(STEPS)}")

    if args.skip_smoke and 1 in steps:
        steps = [s for s in steps if s != 1]
        print("[SKIP] 已跳过第 1 步（--skip-smoke）")

    data_path = (REPO_ROOT / args.data).resolve()
    spec_dir = REPO_ROOT / "results" / "ua_causal_specification"
    adj_dir = REPO_ROOT / "results" / "ua_causal_adjustment"
    fig_dir = REPO_ROOT / "results" / "ua_figure5hi"

    if 1 not in steps:
        if not data_path.exists():
            print(f"ERROR: 找不到数据文件 {data_path}")
            print("       若要用合成数据先跑通流程，请去掉 --skip-smoke。")
            sys.exit(1)
        print(f"[INFO] 使用数据：{data_path}")

    print("将执行以下步骤：")
    for s in steps:
        print(f"  {s}. {STEPS[s]}")

    # ---- 执行
    if 1 in steps:
        run_script("ua_make_example_data.py",
                   ["--out", args.data, "--seed", args.seed])
    if 2 in steps:
        run_script("ua_causal_specification.py",
                   ["--out", "results/ua_causal_specification"])
    if 3 in steps:
        run_script("ua_run_causal_adjustment.py",
                   ["--data", args.data, "--out", "results/ua_causal_adjustment",
                    "--n-boot", args.n_boot, "--seed", args.seed,
                    "--marker-mode", args.marker_mode])
    if 4 in steps:
        run_script("ua_reproduce_figure5hi.py",
                   ["--data", args.data, "--out", "results/ua_figure5hi",
                    "--n-boot", args.n_boot, "--seed", args.seed,
                    "--marker-mode", args.marker_mode])

    print("\n" + "=" * 68)
    print("管线执行完毕")
    print("=" * 68)

    # ---- 校验
    ok = True
    if not args.no_verify:
        ok, checks = verify_outputs(data_path, spec_dir, adj_dir, fig_dir,
                                    args.n_boot, steps)
        print("\n[verify] 产物校验")
        print_table(checks)
        if not ok:
            print("\n[verify] 存在未通过的校验项，请逐条查看上面的 FAIL 行。")

    # ---- 清单
    if args.manifest:
        targets = [data_path,
                   data_path.with_suffix("").with_suffix(".meta.json")]
        for d in (spec_dir, adj_dir, fig_dir):
            targets += [p for p in d.glob("*") if p.is_file()]
        targets += [REPO_ROOT / "src" / f for f in (
            "ua_data_utils.py", "ua_make_example_data.py",
            "ua_causal_specification.py", "ua_run_causal_adjustment.py",
            "ua_reproduce_figure5hi.py")]
        write_manifest(targets, REPO_ROOT / "results" / "MANIFEST.json")

    print("\n主要产物：")
    print(f"  因果规格       {spec_dir}")
    print(f"  调整阶梯       {adj_dir}")
    print(f"  模型对比       {fig_dir}")

    if not ok:
        sys.exit(3)


if __name__ == "__main__":
    main()
