# SweatUric-AI · 汗液尿酸因果机器学习分析管线

**Causal-assumption-guided ML pipeline for non-invasive sweat uric-acid (UA) monitoring — adapted from the PolyCORE lipid framework.**

本仓库是一个**方法论移植 + 流程验证**阶段的代码库，不是一个已完成的检测系统。它做的事情很窄、很明确：
把 *Nature Sensors* 论文公开的「因果假设优先」建模范式（DAG → 后门调整集 → 特征集 → 分组交叉验证）
从**脂质**场景迁移到**尿酸**场景，并让整条链路在一台干净机器上**可一键复现、可机器校验**。

> ## ⚠️ 三条必须先读到的声明
>
> 1. **本仓库内所有数据均为合成数据**（`data/ua_example_merged_data.csv`，由
>    `src/ua_make_example_data.py` 按生理假设生成，`meta.json` 标记 `"synthetic": true`）。
>    仓库里出现的任何 MAE / R² / 置信区间**只用于证明代码能正确运行，不构成任何检测性能证据**。
>    真实受试者数据不存在、也未采集——**伦理审查批件尚未取得**。
> 2. **硬件部分不在本仓库内**。本仓库只覆盖算法与因果建模，不含器件制备工艺、电化学原始曲线、电路图。
>    文中提到的 PCB 采集板是**设计概念**，不是已制造的实物。
> 3. 本仓库为**参赛代码**，为配合材料初评的盲审要求，**不含团队成员姓名、单位、联系方式**。

---

## 1. 来源与署名（重要）

本仓库的四个核心脚本**改编**自下述论文的官方公开代码仓库，不是从零另起炉灶；
其余文件（`ua_data_utils.py`、`run_ua_pipeline.py`、尿酸专用规格与测试）为本仓库新增。

| 本仓库文件 | 上游对应文件 | 关系 |
|---|---|---|
| `src/ua_causal_specification.py` | `src/causal_specification.py` | 改编：DAG 由脂质 3 任务改为 UA 单任务，新增肾功能块与 pH 效应修饰 |
| `src/ua_run_causal_adjustment.py` | `src/run_causal_adjustment.py` | 改编：调整为 8 级阶梯 S0–S7，区分 confounder / precision / modifier / joint |
| `src/ua_reproduce_figure5hi.py` | `src/reproduce_figure5hi.py` | 改编：模型表由规格文件读取，新增簇 bootstrap、2×2 消融、拆分器敏感性、肾功能分层 |
| `src/ua_make_example_data.py` | `src/make_example_data.py` | 重写：CKD-EPI 2021 eGFR、LOD 截尾、pH 响应项、两档自检 |
| `src/ua_data_utils.py` | — | 新增：dtype/编码/切分/泄漏守卫的单一事实源 |
| `run_ua_pipeline.py` | — | 新增：一键运行 + 24 项机器校验 + sha256 产物清单 |

**上游出处**

> Lasalde-Ramírez J. A., Won C., Ji S., et al.
> *Non-invasive continuous lipid profiling via cofactor-refreshing cascading enzymatic reactions
> and causal machine learning.* **Nature Sensors**, vol. 1, Sep. 2026, pp. 811–823.
> DOI: [10.1038/s44460-026-00117-0](https://doi.org/10.1038/s44460-026-00117-0)
> 官方代码仓库：<https://github.com/SIJIEJI/polycore-lipid-causal>（本地副本对应提交 `6277a9b`，2026-07-14）

**移植的不是化学，是因果推断方法论。** 原论文的化学创新是「辅因子刷新级联酶反应」——
用聚吡咯/ATP 微结构持续为甘油三酯的三酶级联（LIP → GK → G3PO）补给被消耗的 ATP。
**尿酸检测是单步酶促反应，不需要任何辅因子**，所以那条化学路线对本项目没有意义；
可迁移的是它「先把因果假设写成图，再据此确定输入变量与评估协议」的建模范式。
逐条改动与理由见 [`docs/UA_MIGRATION_GUIDE.md`](docs/UA_MIGRATION_GUIDE.md)。

**许可状态说明**：截至本仓库发布时，上游 `polycore-lipid-causal` 未声明 LICENSE 文件
（按默认著作权规则为「保留所有权利」）。本仓库以上述署名方式引用并说明派生关系，
发布仅用于学术交流与竞赛材料提交。**如需在任何其他用途下使用本仓库或其派生代码，
请先联系论文作者取得授权**；我们会在上游明确许可后同步补上 LICENSE。

---

## 2. 目录结构

```text
.
├── run_ua_pipeline.py                  ← 一键跑完整条链路（从这里开始）
├── requirements.txt
├── README.md
├── CITATION.cff
├── docs/
│   ├── UA_MIGRATION_GUIDE.md           ← 脂质 → 尿酸：改了什么、为什么必须改
│   └── (CITATION_VERIFICATION.md 为内部核查记录，不随仓库发布)
├── src/
│   ├── ua_data_utils.py                ← 单一事实源：dtype/编码/切分/泄漏守卫/指标
│   ├── ua_causal_specification.py      ← 「宪法」：22 条边的 DAG、调整集推导、特征集、模型定义
│   ├── ua_run_causal_adjustment.py     ← 调整阶梯 S0–S7：把定性图变成定量曲线
│   ├── ua_reproduce_figure5hi.py       ← 7 模型分组 CV + Bootstrap + 2×2 消融 + 敏感性 + 分层
│   └── ua_make_example_data.py         ← 合成数据生成器（带自检，不是结果生成器）
├── data/
│   ├── ua_merged_data_schema.csv       ← 真实数据采集时的列契约（现在只是契约，没有数据）
│   ├── ua_example_merged_data.csv      ← 合成数据（50 人 × 3 时间点 = 150 行）
│   └── ua_example_merged_data.meta.json← "synthetic": true 标记
└── results/                            ← 已提交的运行产物（28 个文件 + MANIFEST.json）
    ├── ua_causal_specification/         DAG 图与边表、调整集推导、特征集、模型定义
    ├── ua_causal_adjustment/            调整阶梯表 + adjustment_ladder.png
    └── ua_figure5hi/                    主结果、折指标、Bootstrap CI、消融、敏感性、分层
```

`results/` 与上游做法不同（上游把 `results/` 排除在版本控制外）。这里**故意提交**，
原因是技术报告正文直接引用这些图与表；把它们放进仓库，报告里的每个数字都能被点到文件、
并被 `MANIFEST.json` 的 sha256 校验到。重新运行会原地覆盖，不会产生分叉副本。

---

## 3. 安装与运行

需要 Python ≥ 3.10（本项目在 3.11 上验证）。

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      macOS/Linux:  source .venv/bin/activate
python -m pip install -r requirements.txt
python run_ua_pipeline.py --manifest
```

一条命令依次执行 4 个步骤，结束时打印 `管线执行完毕`、逐条列出校验结果，
全部通过则以退出码 0 收尾（任一 FAIL 退出码 3）。常用开关：
`--steps 3,4` 只跑部分步骤，`--skip-smoke` 保留已有数据不重新生成，
`--data` 指定数据文件（接真实数据时用），`--no-verify` 跳过校验，
`--manifest` 写出 `results/MANIFEST.json`，`--marker-mode sensor` 用电极原始响应替代酶法参考值。

分步执行（与上面 4 步一一对应，便于单独调试）：

```bash
python src/ua_make_example_data.py     --out data/ua_example_merged_data.csv
python src/ua_causal_specification.py  --out results/ua_causal_specification
python src/ua_run_causal_adjustment.py --data data/ua_example_merged_data.csv --out results/ua_causal_adjustment
python src/ua_reproduce_figure5hi.py   --data data/ua_example_merged_data.csv --out results/ua_figure5hi
```

也可以只跑其中一步（例如只看 DAG 是否无环）：`python src/ua_causal_specification.py --out results/ua_causal_specification`。

**运行后的机器校验**（`verify_outputs()`，共 24 项）逐条打印 PASS/FAIL，覆盖：
DAG 无环、7 个模型 × 恰好 5 折、50 名受试者在训练/测试折之间零重叠、
`Causal_ML` 特征集与 `Ridge` 特征集互异、`bootstrap_ci.csv` 恰为 14 行（7 模型 × 2 指标）、
肾功能分层表里既有 n≥15 的层也有非空 `skip_reason`、合成数据相关性落在 0.15–0.55 的合理区间。

`--manifest` 额外写出 `results/MANIFEST.json`，为全部产物**以及 5 个 `src/*.py` 源文件**记 sha256——
这样报告引用的每张图和每个数字都能反查到产生它的那份代码；改了代码而忘了重跑，清单就会对不上。

唯一的例外是 `data/ua_example_merged_data.meta.json`：它含 `created_utc` 生成时刻，
每次重跑必然变化。其余 27 条校验值已验证可复现——把本仓库克隆到全新目录后直接
`python run_ua_pipeline.py --manifest`，24 项机检全部通过，产物与仓库中提交的文件逐字节相同。

**可复现性**：随机种子固定为 42，且 `RandomForestRegressor(n_jobs=1)`。
后者不是性能取舍——多线程浮点归约顺序不固定，曾导致同一份代码两次运行输出的
CSV 数值差异在 1e-16 量级却 sha256 不一致，无法做产物校验。改回单线程后
**两次完整运行的产物字节级一致**。

所有文本产物（CSV / JSON / 清单）以 LF 换行写出（pandas 默认用 `os.linesep`，
在 Windows 上会写 CRLF，使同一份代码跨平台产物的 sha256 不同）。
配合 `.gitattributes` 的 `eol=lf`，Git 中的字节、磁盘上的字节与
`MANIFEST.json` 里的 sha256 三者始终一致。

---

## 4. 因果建模思路（读代码的顺序）

1. **先写图，再写模型。** `ua_causal_specification.py` 里 DAG 是 15 节点 / 22 条边的显式列表，
   `validate_dag()` 用 Kahn 拓扑排序强制无环，任何环路直接抛错。
2. **最小充分调整集由图推出，不由 p 值推出。** 后门准则逐路径判定，
   13 个候选变量各自有 `_VERDICT_NOTES` 记录「进/出调整集」的理由（混杂 / 精度变量 / 效应修饰 / 联合处理 / 过度调整）。
3. **eGFR 的处置是本项目的关键分歧点。** 在当前图里 eGFR 只有指向 `blood_UA` 的出边、
   没有指向 `sweat_UA` 的边，因此它**不是混杂，而是精度变量**；
   若为了「像混杂」而补一条 `blood_UA → eGFR` 就会成环。
   解决办法是给 eGFR 换身份：**入组基线值**，时间上先于研究期内所有血尿酸测量。
   这样既保持无环，又有生理学依据（反映入组前的累积尿酸负荷）。
   它因此被标为 **Tier-2**，在报告与代码注释中都不被称为混杂因素。
4. **调整阶梯把定性图变成定量曲线。** `ua_run_causal_adjustment.py` 沿 S0→S7 逐级加入变量，
   每级报告偏相关 r 与**簇稳健**置信区间（有效样本量取**受试者人数**而非行数，避免重复测量伪关联）。
5. **评估协议按受试者分组。** `GroupKFold` 而非 `KFold(shuffle=True)`，
   标准化放在 Pipeline 内做折内 fit，`oof_predict()` 内置泄漏守卫
   （检测到同一受试者同时出现在训练/测试折即报错；唯一例外是
   `kfold_optimism_delta`——它**故意**制造泄漏来量化这种乐观偏差）。

---

## 5. 当前运行结果（合成数据，只说明代码行为）

`results/` 中的数字来自随机种子 42，重跑可字节级复现。**它们不是性能指标。**

| 观察 | 数值 | 含义 |
|---|---|---|
| 原始汗液–血尿酸关联 | r = 0.516 [0.352, 0.656] | S0 基线 |
| 加入 Tier-1 最小集（BMI, sex, age） | r → 0.329 | 关联**下降**：被控变量在合成机制里确实共享，属正常偏相关行为 |
| 再加 eGFR（Tier-2 精度变量） | r → 0.291 | 同上；这条与早期草稿「加入 eGFR 后关联增强」的说法相反，已在报告中更正 |
| 因果引导特征集 vs 全特征 | ΔR² = +0.062 | 2×2 消融中「特征集」主效应 |
| RandomForest vs Ridge | ΔR² = −0.061 | 小样本线性机制下更强模型并不更优，合理 |
| `Causal_ML` vs `Ridge` | ΔR² = +0.124 [+0.028, +0.251] | CI 不含 0 |
| `Causal_ML` vs 两个阴性对照 | ΔR² = +0.069 / +0.071，CI **含 0** | **因此「因果引导优于随机特征」这一点尚未被证实**，报告按未证实处理 |
| 拆分器敏感性（分组 vs 打乱） | ΔR² = +0.108 [0.020, 0.234] / +0.244 [0.097, 0.483] | 不按受试者分组会显著高估性能 |

生成器**没有为了故事好看而调参**：它只编码生理假设（CKD-EPI 2021 无种族项 eGFR、
试剂 LOD 8.0 μmol/L 截尾、汗液 pH 对响应的乘性调制），并内置两档自检——
若合成数据变得「太容易」（相关 > 0.70 或 R² > 0.95）会以非 0 退出码报警。
个体响应曲线的斜率在合成数据里完全相同这一事实也被保留展示，而不是被修饰掉。

---

## 6. 证据分级图例

技术报告与本仓库的产物按同一套标记：

- **【已完成·物证】** 有实物/原始文件可指（本仓库中 = 代码、运行产物、`MANIFEST.json`）。
- **【进行中】** 已在做但未闭合（如真实数据采集方案，等待伦理批件）。
- **【计划】** 设计概念，尚无实物（如 PCB 采集板、临床验证）。

## 7. 下一步（与报告第 12 章对应，均为【计划】）

1. 取伦理批件 → 按 `data/ua_merged_data_schema.csv` 的契约采集配对汗液/血尿酸数据；
   代码侧已就绪，真实数据放入 `data/ua_merged_data.csv` 即可复用全部分析（该文件已被 `.gitignore` 禁止入库）。
2. 器件端补齐恒定电位与温度控制，把 `sweat_pH` / `sweat_rate` 从「问卷回忆值」变成同步实测值。
3. 用真实数据重跑本报告的全部分析，并按 Bland–Altman 与临床分界点（男性 420 μmol/L）重新定义评价指标——
   届时 `results/` 里的合成数据产物会被替换，报告也将不再引用它们。

---

如果本仓库对你有帮助，请同时引用上游论文与官方代码仓库（见第 1 节与 `CITATION.cff`）。
