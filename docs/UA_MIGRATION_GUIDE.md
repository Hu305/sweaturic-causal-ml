# UA Causal ML 迁移指南

本文档说明如何把因果机器学习框架从脂质场景的建模思路应用到**尿酸（UA）**场景，以及迁移过程中做了哪些**非照搬**的结构性改动。

> 一句话总结：**核心是可迁移的因果推断方法论，而非特定化学体系。**
> 尿酸检测（尿酸酶法或铜基直接氧化）是**单步反应，不需要任何辅因子**。
> 本管线采用「先把因果假设写成图，再据此确定输入变量与评估协议」的建模范式。

---

## 1. 脂质 vs 尿酸：场景差异决定结构改动

| 维度 | 脂质场景 | 尿酸（本项目） | 为什么必须改 |
|------|---------------|---------------|-------------|
| **核心混杂来源** | 体成分与代谢状态（BMI 等） | **肾功能（eGFR / 肌酐 / 尿素氮）** | 血尿酸约 **70% 经肾脏排泄**，肾功能是压倒性的决定因素 |
| **效应修饰变量** | 无显著项 | **汗液 pH** | 尿酸是弱酸（pKa₁≈5.4），汗液 pH 4.5–7.0 直接改变离子化比例与电极响应 |
| **任务数** | CH + TG 双任务 | UA 单任务 | — |
| **化学体系** | PPy/ATP 辅因子刷新 + 三酶级联 | 镀铜 LIG 低电位**直接氧化** | 尿酸单步反应无需辅因子，从原理上消除辅因子耗尽这一失效模式 |
| **膳食挑战** | 高脂餐、混合餐、高蛋白 | **高嘌呤餐 / 高果糖饮料 / 酒精** | 场景对齐 |
| **受试者数** | 24 人 | 目标 n=50（本项目验证方案） | 混杂变量更多，需要更大样本支撑调整 |

---

## 2. 文件对应关系

| 参考实现 | 本仓库文件 | 改动内容 |
|-----------|-----------|---------|
| `causal_specification.py` | `src/ua_causal_specification.py` | DAG 19 → **22 条边**；新增肾功能块、pH 效应修饰、最小调整集推导、`rationale` 列 |
| `run_causal_adjustment.py` | `src/ua_run_causal_adjustment.py` | 调整阶梯与 DAG 推导对齐；新增 `adjustment_ladder.png`；受试者簇 bootstrap CI |
| `reproduce_figure5hi.py` | `src/ua_reproduce_figure5hi.py` | 7 模型；新增 2×2 消融、折划分敏感性、肾功能分层、个体响应曲线；标准化移入折内 |
| `make_example_data.py` | `src/ua_make_example_data.py` | 移除确定性反演公式；eGFR 改由 CKD-EPI 从肌酐导出；新增 LOD 左删失、T0/T1/T2 阶段、`meta.json` 合成标记、生成后自检 |
| `merged_data_schema.csv` | `data/ua_merged_data_schema.csv` | 血尿酸单位改为 **μmol/L**（与指南阈值 420/360 同单位）；新增 `column_role` 列标注每个变量在因果模型中的身份 |
| （无） | `src/ua_data_utils.py` | **新增**：性别编码、肾功能分层、统一加载、无泄漏 CV、簇 bootstrap 的单一事实源 |
| （无） | `run_ua_pipeline.py` | 一键运行 + 产物校验 + `--manifest`（sha256 清单） |

---

## 3. DAG 的 3 处结构性改动

原 19 条边的版本存在**两个结构漏洞**，本次一并补齐（19 → 22）：

1. **`sweat_rate` 没有任何父节点**。它在原图里被当作外生变量，因此在数学上无法对它做后门调整，也就不可能把它当作第二个处理变量来联合建模。补 `BMI → sweat_rate`、`sex → sweat_rate`。
2. **`sweat_UA_sensor` 是悬空节点**（只有入边、无出边、无对应数据列，也没有任何模型引用它）。补 `sweat_UA → sweat_UA_sensor` 这条**测量边**，使「电极响应是汗液尿酸浓度的观测」这一语义落地，部署模式下才能真正用上它。
3. **新增肾功能块**：`eGFR / creatinine / BUN → blood_UA`，以及 `age → eGFR`、`sex → eGFR`。

---

## 4. 关于 eGFR：为什么它是「精度变量」而不是「混杂变量」

这一条最容易被评审挑错，必须讲准。

在后门准则的意义下，混杂变量需要**同时**是处理变量与结局的父节点。而在本项目 DAG 中，eGFR **只有出边指向 `blood_UA`**，没有任何边指向 `sweat_UA`——所以严格来说它是**结局的父节点**，属于**精度变量（precision variable）**，不是混杂变量。

有一种常见但不成立的做法是补一条 `blood_UA → eGFR` 把它"论证成"混杂。这会**构成环**：

```
eGFR → blood_UA → eGFR     ← 破坏有向无环性，图不再是 DAG
```

本管线里 `validate_dag()` 会用拓扑排序自动检测环（见 `results/ua_causal_specification/dag_validation.csv` 的 `is_acyclic` 列），所以这类错误不会被口头承诺掩盖过去。

**正确做法**（本项目采用）：

> eGFR 取**入组基线值**（入组时的一次检测），在时间上先于本研究期间的所有 `blood_UA` 测量。基线 eGFR 代表受试者**入组前已累积的尿酸盐负荷与肾脏排泄能力**。

这样它作为「既有肾功能状态」的代理进入调整集（Tier-2，在 Tier-1 最小集之上追加），既不构成环，又能提高估计精度、并在肾功能受损人群中维持映射稳定性。但报告与代码都**不把它称为混杂变量**。

---

## 5. 最小调整集是怎么推导出来的

对处理变量 `sweat_UA` 的每个父节点 `p`，判断是否存在一条 `p → … → blood_UA` 且**不经过处理变量**的有向路径：

- 存在 → `p` 是混杂变量，必须调整
- 不存在 → `p` 不是混杂变量

推导结果（完整表见 `results/ua_causal_specification/adjustment_set_derivation.csv`）：

| 候选变量 | 是处理的父节点 | 有通路到结局 | 判定 | 进入哪一层 |
|---|---|---|---|---|
| BMI | 是 | 是 | **混杂变量** | Tier-1 |
| sex | 是 | 是（另经 eGFR） | **混杂变量** | Tier-1 |
| age | 是 | 是（另经 eGFR） | **混杂变量** | Tier-1 |
| sweat_rate | 是 | **否** | 第二处理变量（不是混杂） | Tier-1（作为 T2） |
| sweat_pH | 是 | **否** | **效应修饰变量** | 不进调整集，进交互 |
| blood_UA | 是 | — | 结局本身（因果通路） | 不调整 |
| eGFR | 否 | 是 | 精度变量 | Tier-2 |
| creatinine / BUN | 否 | 是 | 与 eGFR 共线 | 不纳入 |
| fat_mass / muscle_mass / BMR / blood_glucose | 否 | 是 | 对后门准则无额外贡献 | 仅 full_model 对照 |

**Tier-1 严格后门最小集 = `sweat_UA; sweat_rate; BMI; sex; age`（5 列）**
**Tier-2 = Tier-1 + `eGFR`（6 列）**

---

## 6. 七个模型

```
Simple                   : sweat_UA                                     ← 单标记基线（对应论文 Simple）
Multi                    : sweat_UA + sweat_rate                        ← 双变量基线（稀释控制）
Ridge                    : 全 10 特征 + L2                              ← 多特征回归上界对照
Lasso                    : 全 10 特征 + L1                              ← 稀疏选择对照
Causal_ML                : Tier-2 最小集（6 列）+ L2                    ← 核心实验
Negative_Control         : 协变量 + pH，无任何汗液动态信号（8 列）      ← 静态负对照
Negative_Control_Dynamic : 上面再加 sweat_rate（9 列）                  ← 动态负对照
```

**关键修复**：原版 `Causal_ML` 与 `Ridge` 的特征列表**完全相同**（都是 10 列），
所以「因果增益」根本无法与「多特征增益」分离。本版 `Causal_ML` 只有 6 列，
**故意比 `Ridge` 小**——这正是因果方法的关键：用后门准则把不需要控制的变量剔出去，
而不是把能拿到的变量全丢给模型靠正则化去压。

---

## 7. 关键设计决策

### 为什么用 `GroupKFold` 而不是 `KFold`？
同一受试者有多次重复测量。若同一人同时出现在训练折与测试折，模型可以靠"记住这个人"而不是"学会这个规律"拿到高分。常见做法是用 `KFold`，但本管线用 `GroupKFold`，并额外产出 `sensitivity_splitter.csv` **量化**这一乐观偏差。

### 为什么标准化必须在折内做？
原版 `StandardScaler().fit_transform(X)` 在 CV 循环**之外**执行，测试折的均值/方差已经漏进训练折。本版把 scaler 放进 `Pipeline`，由 `oof_predict()` 保证每折独立 `fit`。

### 为什么 `Causal_ML` 用 Ridge 而不是随机森林？
BMI、体脂、肌肉量、BMR 高度共线。Ridge（L2）比 Lasso（L1）在共线条件下更稳定（alpha=1.0）。随机森林只在 **2×2 消融**里作为估计器对照出现，用来回答"换个更强模型会不会更好"。

### 为什么 Bootstrap 要按受试者整簇重采样？
同一人的 3 次测量高度相关。按测量条重采样等于假装样本量是 150 而不是 50，会系统性把置信区间压窄（伪重复 / pseudoreplication）。`cluster_bootstrap_ci()` 按人整簇重采样；调整阶梯的 p 值也按 Fisher z 用**受试者数**作为有效样本量。

### 为什么合成数据要带 `meta.json` 标记？
`ua_make_example_data.py` 原本有一条确定性反演公式（先有血尿酸再"算出"汗尿酸），造出的数据任何模型都能拿到接近 1 的 R²。这种数字一旦被当成性能证据，性质就是数据造假。现在生成器只编码生理假设、不设定结果目标，并以 `meta.json` 的 `synthetic: true` 标记驱动下游强制打「合成数据」水印；生成后自检若发现 `corr(sweat_UA, blood_UA) > 0.55` 或 Ridge 五折 R² > 0.85，会直接以非 0 退出码报错。

---

## 8. 使用真实数据的步骤

1. **整理数据格式**：CSV 需包含 `data/ua_merged_data_schema.csv` 中列出的列。血尿酸单位若为 mg/dL，加 `--blood-unit mg/dL`（或用 `auto` 自动判断）。
2. **替换数据文件**：例如放到 `data/merged_data.csv`。
3. **运行**：
   ```bash
   python run_ua_pipeline.py --data data/merged_data.csv --skip-smoke --manifest
   ```
4. **查看产物**：
   - `results/ua_causal_adjustment/adjustment_ladder.png` —— 加入 eGFR 后关联是否增强（检验"肾功能是核心混杂"）
   - `results/ua_figure5hi/figure5hi.png` —— 7 模型对比
   - `results/ua_figure5hi/paired_delta_ci.csv` —— `Causal_ML` vs 负对照的配对差值是否显著
   - `results/ua_figure5hi/ablation_heatmap.png` —— 特征集增益 vs 估计器增益
   - `results/ua_figure5hi/sensitivity_splitter.csv` —— 受试者泄漏的乐观偏差
   - `results/ua_figure5hi/renal_stratified_metrics.csv` —— 跨肾功能稳健性
   - `results/ua_figure5hi/individual_response_curves.png` —— 个体化映射斜率
   - `results/MANIFEST.json` —— 全部产物的 sha256 校验值

---

## 9. 命令速查

```bash
python run_ua_pipeline.py                        # 全流程（合成数据）
python run_ua_pipeline.py --steps 2              # 只重写因果规格
python run_ua_pipeline.py --steps 3,4 --n-boot 200   # 只重跑诊断与模型对比，快速看结果
python run_ua_pipeline.py --marker-mode sensor   # 用"电极响应"替代酶法汗液尿酸
python run_ua_pipeline.py --manifest             # 附 sha256 清单
python run_ua_pipeline.py --skip-smoke --data data/merged_data.csv --blood-unit mg/dL
```
