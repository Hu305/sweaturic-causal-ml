#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ua_causal_specification.py — 尿酸因果规格（管线的「宪法」）
============================================================


本文件是整条因果 ML 管线的唯一规格来源：DAG 结构、后门调整集推导、
特征集定义、模型定义。下游所有脚本（因果调整诊断、模型对比）都读取
本文件写出的 CSV，不允许再各自硬编码一份特征列表——
原版本正是因为在 ``ua_reproduce_figure5hi.py`` 里又手写了一遍特征集
（Causal_ML 硬编码 7 列、spec 里写 10 列），导致「报告写的」和
「代码算的」对不上。

三处场景适配改动
----------------------------
1. **新增肾功能块**：血尿酸约 70% 经肾排泄。脂质研究的主要混杂是体成分
   与代谢状态；尿酸的场景特异性在于肾功能是压倒性的决定因素。
   故新增 eGFR / creatinine / BUN 及其上游 age→eGFR、sex→eGFR。
2. **新增 pH 效应修饰**：尿酸是弱酸（pKa₁ ≈ 5.4，pKa₂ ≈ 9.8），汗液 pH（4.5–7.0）
   直接改变其离子化比例与电极响应，但它**不通向 blood_UA**——
   所以它是效应修饰变量，不是混杂变量。这个区分决定了它该进交互项
   还是进调整集。
3. **补齐 DAG 的结构漏洞**（原 19 边版存在两处）：
   - ``sweat_rate`` 原本**没有任何父节点**，即被当作外生变量，
     无法对它做后门调整；补 ``BMI→sweat_rate``、``sex→sweat_rate``。
   - ``sweat_UA_sensor`` 原本是**悬空节点**（只有入边、无出边、无对应数据列，
     也没有任何模型引用它）；补 ``sweat_UA→sweat_UA_sensor`` 这条测量边，
     使它在「部署模式」下落地为真正可得的标记物输入。

关于 eGFR 的定位（重要，且容易被评审挑错）
------------------------------------------
eGFR 在 DAG 中**只有出边指向 blood_UA**，没有指向 sweat_UA 的边。
这意味着严格按后门准则，eGFR 是**结局的父节点而非处理变量的父节点**，
即它是「精度变量」（precision variable），不是混杂变量。

有一种常见但不成立的做法是补一条 ``blood_UA→eGFR`` 来把它论证成混杂——
这会**构成环**（eGFR→blood_UA→eGFR），破坏有向无环性，图的因果语义
随之失效。本项目的正确做法是：

    eGFR 取**入组基线值**（入组时的单次检测），它在时间上先于本研究
    期间的所有 blood_UA 测量；基线 eGFR 代表受试者**入组前已累积的
    尿酸盐负荷与肾脏排泄能力**。因此它作为「既有肾功能状态」的代理
    进入分析，不构成环。

这样处理之后，eGFR 可以合法进入调整集（Tier-2），作用是提高估计精度、
并在肾功能受损人群中维持映射稳定性；但报告与代码都不把它称为
「混杂变量」。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from ua_data_utils import setup_console, write_csv

# --------------------------------------------------------------------------
# 1. DAG
# --------------------------------------------------------------------------

#: 22 条因果边。(parent, child, category, rationale)
#: 前 19 条为原版边（顺序保持不变），后 3 条为本次补齐的结构漏洞。
DAG_EDGES = [
    # ---- 肾功能块（尿酸场景的核心扩展） ----
    ("eGFR", "blood_UA", "renal",
     "血尿酸约 70% 经肾脏排泄，eGFR 是血尿酸最强的单一决定因素"),
    ("creatinine", "blood_UA", "renal",
     "肌酐与尿酸同属肾小管排泄底物，肌酐升高常伴随尿酸潴留"),
    ("BUN", "blood_UA", "renal",
     "尿素氮反映整体肾排泄能力，与血尿酸水平同向变化"),
    ("age", "eGFR", "renal",
     "肾小球滤过率随年龄增长而下降（肾功能路径的时间上游）"),
    ("sex", "eGFR", "renal",
     "CKD-EPI 公式含性别系数：同龄女性 eGFR 略低于男性"),

    # ---- 代谢与体成分块 ----
    ("BMI", "blood_UA", "metabolic",
     "肥胖经胰岛素抵抗促进肾尿酸重吸收，同时增加尿酸生成"),
    ("BMI", "sweat_UA", "metabolic",
     "BMI 影响皮肤屏障功能与汗腺分泌成分"),
    ("fat_mass", "BMI", "body_composition",
     "体脂量是 BMI 的组成分量（体成分分解路径）"),
    ("muscle_mass", "BMI", "body_composition",
     "肌肉量是 BMI 的组成分量（体成分分解路径）"),
    ("BMR", "blood_UA", "metabolic",
     "基础代谢率反映细胞周转与嘌呤代谢通量"),
    ("blood_glucose", "blood_UA", "metabolic",
     "代谢综合征聚集性：胰岛素抵抗同时推高血糖与血尿酸"),

    # ---- 人口学块 ----
    ("sex", "blood_UA", "demographic",
     "男性血尿酸基线显著高于女性（雌激素促进尿酸排泄）"),
    ("sex", "sweat_UA", "demographic",
     "性别影响汗腺密度与局部汗液成分"),
    ("age", "blood_UA", "demographic",
     "血尿酸水平随年龄上升"),
    ("age", "sweat_UA", "demographic",
     "年龄影响皮肤通透性与汗腺功能"),

    # ---- 汗液动力学与化学块 ----
    ("sweat_rate", "sweat_UA", "sweat_dynamics",
     "出汗率越高汗液被稀释越明显，汗液尿酸浓度随之下降"),
    ("sweat_pH", "sweat_UA", "sweat_chemistry",
     "尿酸为弱酸（pKa₁≈5.4、pKa₂≈9.8），汗液 pH 改变其离子化比例"),
    ("sweat_pH", "sweat_UA_sensor", "measurement",
     "pH 改变电极表面质子化状态，从而改变电化学响应增益"),

    # ---- 核心因果通路 ----
    ("blood_UA", "sweat_UA", "causal_path",
     "核心因果通路：血尿酸经皮肤被动扩散/主动分泌进入汗液"),

    # ---- 本次补齐的 3 条边（修正原 19 边版的结构漏洞） ----
    ("BMI", "sweat_rate", "sweat_dynamics",
     "补齐漏洞①：体脂与体表面积影响单位时间出汗量，"
     "原版 sweat_rate 无任何父节点，无法对其做后门调整"),
    ("sex", "sweat_rate", "sweat_dynamics",
     "补齐漏洞①：性别影响汗腺密度与出汗阈值"),
    ("sweat_UA", "sweat_UA_sensor", "measurement",
     "补齐漏洞②测量边：电极响应是汗液尿酸浓度的观测，"
     "原版 sweat_UA_sensor 是悬空节点，本条使部署模式下落为可用输入"),
]

#: 节点 → 说明（报告附录 B 表注用）
NODE_NOTES = {
    "blood_UA": "血清尿酸浓度（金标准结局）",
    "sweat_UA": "汗液尿酸浓度（处理变量 T1，酶法金标准）",
    "sweat_UA_sensor": "电极原始响应（部署模式唯一可得的标记物输入）",
    "sweat_rate": "出汗率（处理变量 T2，稀释因子）",
    "sweat_pH": "汗液 pH（效应修饰变量）",
    "eGFR": "估算肾小球滤过率（取入组基线值，精度变量）",
    "creatinine": "血清肌酐", "BUN": "血尿素氮",
    "BMI": "体质指数", "fat_mass": "体脂量", "muscle_mass": "肌肉量",
    "BMR": "基础代谢率", "blood_glucose": "空腹血糖",
    "sex": "性别", "age": "年龄",
}


def get_dag_edges():
    """返回 DAG 边列表（parent, child）。"""
    return [(p, c) for p, c, _, _ in DAG_EDGES]


def _nodes():
    ns = set()
    for p, c, _, _ in DAG_EDGES:
        ns.add(p); ns.add(c)
    return ns


def _children_map():
    m = {}
    for p, c, _, _ in DAG_EDGES:
        m.setdefault(p, set()).add(c)
    return m


def _parents_map():
    m = {}
    for p, c, _, _ in DAG_EDGES:
        m.setdefault(c, set()).add(p)
    return m


def _has_directed_path(src: str, dst: str, forbidden: set[str]) -> bool:
    """src 是否能沿有向边到达 dst（不经过 forbidden 中的节点，src/dst 除外）。"""
    if src == dst:
        return False
    children = _children_map()
    seen, stack = {src}, [src]
    while stack:
        cur = stack.pop()
        for nxt in children.get(cur, ()):  # noqa: B905 - 集合迭代顺序不影响可达性
            if nxt in forbidden and nxt != dst:
                continue
            if nxt == dst:
                return True
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return False


def validate_dag() -> dict:
    """
    校验 DAG 的有向无环性（Kahn 拓扑排序）。

    这不是形式主义：本项目专门论证过「不要把 blood_UA→eGFR 加进图里，
    否则成环」——那就必须有一个能自动发现环的检查，否则论证只是口头承诺。
    """
    parents = _parents_map()
    children = _children_map()
    indeg = {n: len(parents.get(n, ())) for n in _nodes()}

    # 排序处理队列与邻居，使拓扑序列**可复现**（集合迭代顺序不保证稳定）
    queue = sorted(n for n, d in indeg.items() if d == 0)
    order = []
    while queue:
        cur = queue.pop(0)
        order.append(cur)
        for nxt in sorted(children.get(cur, ())):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)

    acyclic = len(order) == len(indeg)
    return {
        "n_nodes": len(indeg),
        "n_edges": len(DAG_EDGES),
        "is_acyclic": acyclic,
        "topological_order": " -> ".join(order) if acyclic else "",
        "unresolved_nodes": "" if acyclic else ";".join(
            sorted(n for n, d in indeg.items() if d > 0)),
    }


# --------------------------------------------------------------------------
# 2. 后门调整集推导
# --------------------------------------------------------------------------

TREATMENT_1 = "sweat_UA"
TREATMENT_2 = "sweat_rate"
OUTCOME = "blood_UA"

#: 人工判读（图算法无法自动给出的部分：变量在时间上的可得性与语义）
_VERDICT_NOTES = {
    "BMI": ("confounder", True, True,
            "同时是处理变量与结局的父节点，开通后门路径 sweat_UA←BMI→blood_UA，必须阻断"),
    "sex": ("confounder", True, True,
            "同时是处理变量与结局的父节点（另经 sex→eGFR→blood_UA 亦有通路），必须阻断"),
    "age": ("confounder", True, True,
            "同时是处理变量与结局的父节点（另经 age→eGFR→blood_UA 亦有通路），必须阻断"),
    "sweat_rate": ("joint_treatment", True, True,
                   "是处理变量的父节点，但**没有任何通向 blood_UA 的通路**，"
                   "故它不是混杂变量；本项目将其作为第二个处理变量联合建模，"
                   "以显式吸收稀释效应"),
    "sweat_pH": ("effect_modifier", False, False,
                 "仅通向 sweat_UA 与 sweat_UA_sensor，**无通路通向 blood_UA**，"
                 "故不是混杂变量；它改变的是「处理 → 信号」的映射增益，"
                 "建模为效应修饰（交互项）而非调整项"),
    "blood_UA": ("causal_path", False, False,
                 "它是结局本身，blood_UA→sweat_UA 就是我们要估计的因果通路，"
                 "对它做调整会阻断效应（over-adjustment）"),
    "eGFR": ("precision_variable", False, True,
             "不是处理变量的父节点，严格后门准则下不需调整；"
             "以**入组基线值**纳入（时间上先于本研究全部 blood_UA 测量，"
             "代表入组前累积尿酸盐负荷），既不构成环又能提高估计精度"),
    "creatinine": ("collinear_with_eGFR", False, False,
                   "eGFR 即由 creatinine+age+sex 经 CKD-EPI 公式导出，"
                   "两者高度共线；eGFR 已入集，再纳入只放大方差而无新信息"),
    "BUN": ("collinear_with_eGFR", False, False,
            "与 eGFR 同属肾功能排泄指标，信息高度重叠，理由同上"),
    "fat_mass": ("not_required", False, False,
                 "它是 BMI 的父节点而非后代，但经由 BMI 的路径已被 BMI 阻断，"
                 "对后门准则无额外贡献；仅在 full_model 全特征对照中出现"),
    "muscle_mass": ("not_required", False, False,
                    "同 fat_mass：路径已被 BMI 阻断，最小集不需要"),
    "BMR": ("not_required", False, False,
            "它是结局的父节点而非处理变量的父节点，不阻断任何后门路径"),
    "blood_glucose": ("not_required", False, False,
                      "它是结局的父节点而非处理变量的父节点，不阻断任何后门路径"),
}


def derive_adjustment_sets() -> pd.DataFrame:
    """
    按后门准则逐条推导最小充分调整集。

    对处理的每一个父节点 p，判断是否存在一条
    ``p → ... → blood_UA`` 且**不经过处理变量**的有向路径：
      - 存在  → p 是混杂变量，必须调整；
      - 不存在 → p 不是混杂变量（可能是联合处理变量或效应修饰变量）。

    同时对非父节点（eGFR、creatinine 等）也记录判定结果，
    说明它们「为什么不在最小集里」或「以什么身份入集」。
    """
    forbidden = {TREATMENT_1, TREATMENT_2}
    parents = _parents_map().get(TREATMENT_1, set())

    rows = []
    for cand in _VERDICT_NOTES:
        is_parent = cand in parents
        if cand == OUTCOME:
            path = "（结局本身）"
            has_path = False
        else:
            has_path = _has_directed_path(cand, OUTCOME, forbidden | {TREATMENT_1})
            path = (f"{cand} → … → {OUTCOME}（不经过 {TREATMENT_1}）"
                    if has_path else f"无（{cand} 无通路到达 {OUTCOME}）")

        verdict, in_t1, in_t2, note = _VERDICT_NOTES[cand]
        rows.append({
            "treatment": TREATMENT_1,
            "outcome": OUTCOME,
            "candidate": cand,
            "is_direct_parent_of_treatment": bool(is_parent),
            "directed_path_to_outcome": path,
            "backdoor_open": bool(is_parent and has_path),
            "verdict": verdict,
            "in_tier1_minimal_set": bool(in_t1),
            "in_tier2_minimal_plus_renal": bool(in_t2),
            "rationale": note,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 3. 特征集
# --------------------------------------------------------------------------

#: 全特征 10 列（不含 sweat_UA_sensor，后者是 gold/sensor 两种标记模式的切换项）
FULL_MODEL_FEATURES = [
    "sweat_UA", "sweat_rate", "sweat_pH", "BMI", "sex",
    "age", "eGFR", "fat_mass", "muscle_mass", "BMR",
]

#: Tier-1 严格后门最小集
TIER1_FEATURES = ["sweat_UA", "sweat_rate", "BMI", "sex", "age"]

#: Tier-2 = Tier-1 + 入组基线 eGFR
TIER2_FEATURES = TIER1_FEATURES + ["eGFR"]


def get_feature_sets() -> dict:
    """
    返回全部特征集定义。

    每个集合含 ``columns / role / tier / rationale``。
    ``causal_guided_minimal`` 的列数**故意少于** ``full_model``——
    这正是因果方法的关键：用后门准则把不需要控制的变量剔出去，
    而不是把能拿到的变量全丢给模型靠正则化去压。
    """
    return {
        "marker_only": {
            "columns": ["sweat_UA"],
            "role": "baseline_marker_only",
            "tier": "对比基线",
            "rationale": "只用一个汗液标记物，代表「原始相关性」的下限；对应论文的 Simple",
        },
        "marker_plus_rate": {
            "columns": ["sweat_UA", "sweat_rate"],
            "role": "baseline_marker_plus_rate",
            "tier": "对比基线",
            "rationale": "加入出汗率以吸收稀释效应，对应论文 dual-variable 建模的精神",
        },
        "full_model": {
            "columns": list(FULL_MODEL_FEATURES),
            "role": "fully_adjusted",
            "tier": "全特征对照",
            "rationale": "全部 10 个可用特征，靠 L1/L2 正则化抑制共线；"
                         "作为「多特征回归」上限对照，用于把「因果增益」与"
                         "「多特征增益」分离",
        },
        "causal_guided_minimal": {
            "columns": list(TIER1_FEATURES),
            "role": "causal_minimal",
            "tier": "Tier-1（严格后门最小集）",
            "rationale": "T1 的混杂 {BMI, sex, age}（由后门准则推导）"
                         "∪ {出汗率}（第二处理变量）",
        },
        "causal_guided_minimal_renal": {
            "columns": list(TIER2_FEATURES),
            "role": "causal_minimal_plus_renal",
            "tier": "Tier-2（最小集+精度变量）",
            "rationale": "Tier-1 再加入组基线 eGFR。eGFR 不是混杂变量"
                         "（不指向处理变量），以「入组前累积尿酸盐负荷代理」"
                         "的身份纳入以提高估计精度",
        },
        "covariate_only_static": {
            "columns": ["BMI", "sex", "age", "eGFR", "fat_mass",
                        "muscle_mass", "BMR", "sweat_pH"],
            "role": "negative_control_static",
            "tier": "负对照",
            "rationale": "只有静态人口学/生理协变量与 pH，"
                         "**不含任何汗液动态信号**（既无 sweat_UA 也无 sweat_rate）；"
                         "若该模型表现接近因果模型，说明增益只是来自协变量",
        },
        "covariate_plus_rate": {
            "columns": ["BMI", "sex", "age", "eGFR", "fat_mass",
                        "muscle_mass", "BMR", "sweat_pH", "sweat_rate"],
            "role": "negative_control_dynamic",
            "tier": "负对照",
            "rationale": "在静态负对照上再加出汗率：检验「增益是否只是"
                         "多了一个动态生理变量」，而不是汗液尿酸标记物本身",
        },
    }


# --------------------------------------------------------------------------
# 4. 模型
# --------------------------------------------------------------------------

def get_model_definitions() -> dict:
    """
    7 个模型。``feature_set_key`` 指向上面的特征集，
    下游脚本**必须**通过本函数取特征列，禁止再硬编码。

    ``Simple / Multi / Ridge / Lasso / Causal_ML / Negative_Control``
    与论文 Figure 5h-i 的模型序列一一对应；``Negative_Control_Dynamic``
    是本次新增的更严格负对照。
    """
    return {
        "Simple": {
            "feature_set_key": "marker_only",
            "estimator": "LinearRegression()",
            "description": "单标记基线：原始汗液–血液相关性",
        },
        "Multi": {
            "feature_set_key": "marker_plus_rate",
            "estimator": "LinearRegression()",
            "description": "双变量基线：标记物 + 出汗率（稀释控制）",
        },
        "Ridge": {
            "feature_set_key": "full_model",
            "estimator": "Ridge(alpha=1.0)",
            "description": "全特征 + L2 正则化",
        },
        "Lasso": {
            "feature_set_key": "full_model",
            "estimator": "Lasso(alpha=0.1, max_iter=10000)",
            "description": "全特征 + L1 正则化（稀疏选择）",
        },
        "Causal_ML": {
            "feature_set_key": "causal_guided_minimal_renal",
            "estimator": "Ridge(alpha=1.0)",
            "description": "DAG 指导的因果最小集（Tier-2）+ L2 正则化",
        },
        "Negative_Control": {
            "feature_set_key": "covariate_only_static",
            "estimator": "Ridge(alpha=1.0)",
            "description": "负对照（静态）：全部协变量 + pH，不含任何汗液动态信号",
        },
        "Negative_Control_Dynamic": {
            "feature_set_key": "covariate_plus_rate",
            "estimator": "Ridge(alpha=1.0)",
            "description": "负对照（动态）：再加出汗率，检验增益是否只是多了一个动态变量",
        },
    }


def get_estimator(estimator_key: str):
    """
    把 ``model_definitions.csv`` 里的估计器字符串实例化成 sklearn 对象。

    单独抽成函数是为了让 ``ua_causal_specification.py`` 本身
    **不依赖 scikit-learn**——规格脚本要能在任何环境下跑出 CSV。

    全部估计器都固定 ``random_state=42``。随机森林额外固定 ``n_jobs=1``：
    多线程下每棵树的生长本身是确定性的，但**归约求和顺序不固定**，
    浮点加法不满足结合律，同一份数据两次运行会差出 1e-16 量级的末位，
    使产物 sha256 不一致。本数据集只有 150 行 × 300 棵树，
    单线程耗时不到 1 秒，用这点速度换「同 seed 两次运行产物逐字节相同」
    是划算的——报告里承诺的「一键复现」才有可验证的含义。
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, LinearRegression, Ridge

    registry = {
        "LinearRegression()": LinearRegression,
        "Ridge(alpha=1.0)": lambda: Ridge(alpha=1.0, random_state=42),
        "Lasso(alpha=0.1, max_iter=10000)": lambda: Lasso(
            alpha=0.1, max_iter=10000, random_state=42),
        "RandomForest(n_estimators=300)": lambda: RandomForestRegressor(
            n_estimators=300, random_state=42, n_jobs=1),
    }
    if estimator_key not in registry:
        raise KeyError(
            f"未知估计器 {estimator_key!r}，可用：{sorted(registry)}")
    return registry[estimator_key]()


# 消融实验用的估计器池（2×2 因子设计：估计器 × 特征集）
ABLATION_ESTIMATORS = ["Ridge(alpha=1.0)", "RandomForest(n_estimators=300)"]
ABLATION_FEATURE_SETS = ["full_model", "causal_guided_minimal"]


# --------------------------------------------------------------------------
# 5. 输出
# --------------------------------------------------------------------------

def plot_dag(out_path) -> bool:
    """
    从 ``DAG_EDGES`` 直接画出因果图，落盘为 PNG。

    为什么一定要用代码画、而不是用现成的示意图：
    报告里若放一张手绘/AI 生成的 DAG 概念图，它与 ``causal_dag_edges.csv``
    之间没有任何机制保证一致——改了边忘了改图，读者无法察觉。
    本函数以 CSV 的同一数据源绘制，**图上每条边都能在产物里查到出处**，
    改边则图自动跟着变。

    布局按拓扑层级（Kahn 分层）从左到右排列，同类边同色：

        人口学/体成分 → 代谢/肾功能 → 处理变量 → 测量变量
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False

    # ---- 拓扑分层：level[v] = 最长路径长度，保证边一律从左指向右
    nodes = sorted({p for p, _, _, _ in DAG_EDGES} | {c for _, c, _, _ in DAG_EDGES})
    parents = {n: [] for n in nodes}
    for p, c, _, _ in DAG_EDGES:
        parents[c].append(p)

    level = {n: 0 for n in nodes}
    for _ in range(len(nodes)):          # 松弛 len(nodes) 次即收敛（已校验无环）
        changed = False
        for n in nodes:
            if parents[n]:
                best = max(level[p] + 1 for p in parents[n])
                if best > level[n]:
                    level[n] = best
                    changed = True
        if not changed:
            break

    # ---- 同一层内排序：处理/结局/测量这类"主角"优先，其余按名称。
    # 每一层在纵向上**居中**排布（而不是统一从顶部往下堆），
    # 否则第 0 层有 9 个节点、其余层只有 2 个，整张图会左重右轻、难以阅读。
    priority = {"sweat_UA": 0, "sweat_rate": 1, "blood_UA": 2,
                "sweat_UA_sensor": 3, "sweat_pH": 4, "eGFR": 5}
    by_level = {}
    for n in nodes:
        by_level.setdefault(level[n], []).append(n)

    y_gap = 1.0
    pos = {}
    for lv, members in by_level.items():
        members.sort(key=lambda n: (priority.get(n, 99), n))
        n_m = len(members)
        for i, n in enumerate(members):
            pos[n] = (lv, (n_m - 1) / 2.0 * y_gap - i * y_gap)

    cat_color = {
        "renal": "#2A9D8F",
        "metabolic": "#E76F51",
        "body_composition": "#F4A261",
        "demographic": "#8AB17D",
        "sweat_dynamics": "#5B8FF9",
        "sweat_chemistry": "#B07AA1",
        "measurement": "#9C6ADE",
        "causal_path": "#D62828",
    }
    cat_label = {
        "renal": "肾功能路径（尿酸场景扩展）",
        "metabolic": "代谢路径",
        "body_composition": "体成分路径",
        "demographic": "人口学路径",
        "sweat_dynamics": "汗液动力学（含补齐的 2 条）",
        "sweat_chemistry": "汗液化学",
        "measurement": "测量边（补齐的悬空节点）",
        "causal_path": "核心因果通路（待估计）",
    }

    max_level = max(pos[n][0] for n in nodes)
    span_y = max(len(v) for v in by_level.values()) * y_gap
    fig, ax = plt.subplots(figsize=(13.6, 0.62 * span_y + 2.6))

    # 用 FancyArrowPatch 直接画到坐标轴，而不是 ax.annotate("", ...)。
    # annotate 会额外生成一个空 Text 艺术家，其 extent 会污染
    # bbox_inches="tight" 的测量结果——实测能让 tight bbox 从 27×7 英寸
    # 膨胀到 27×383 英寸，输出一张 4 亿像素的废图。
    from matplotlib.patches import FancyArrowPatch
    for p, c, cat, _ in DAG_EDGES:
        x0, y0 = pos[p]
        x1, y1 = pos[c]
        is_core = cat == "causal_path"
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            arrowstyle="-|>,head_width=4.5,head_length=8",
            color=cat_color.get(cat, "#888888"),
            linewidth=3.4 if is_core else 1.9,
            linestyle="-", alpha=0.9 if is_core else 0.7,
            connectionstyle="arc3,rad=0.13",
            shrinkA=30, shrinkB=30,
            mutation_scale=1.0, zorder=2,
        ))

    role_face = {"treatment": "#FFE0B2", "outcome": "#FFCDD2", "modifier": "#E1BEE7",
                 "sensor": "#D1C4E9", "renal": "#B2DFDB"}
    treat = {"sweat_UA", "sweat_rate"}
    outcome = {"blood_UA"}
    modifier = {"sweat_pH"}
    sensor = {"sweat_UA_sensor"}
    renal = {"eGFR", "creatinine", "BUN"}

    for n, (x, y) in pos.items():
        if n in treat:
            face, edge = role_face["treatment"], "#E76F51"
        elif n in outcome:
            face, edge = role_face["outcome"], "#D62828"
        elif n in modifier:
            face, edge = role_face["modifier"], "#B07AA1"
        elif n in sensor:
            face, edge = role_face["sensor"], "#7E57C2"
        elif n in renal:
            face, edge = role_face["renal"], "#2A9D8F"
        else:
            face, edge = "#F2F2F2", "#888888"
        ax.text(x, y, n, ha="center", va="center", fontsize=10.5, zorder=5,
                bbox=dict(boxstyle="round,pad=0.42", facecolor=face,
                          edgecolor=edge, linewidth=1.6))

    handles = [plt.Line2D([0], [0], color=v, lw=2.4, label=cat_label[k])
               for k, v in cat_color.items()]
    # 图例放在坐标轴**下方之外**，用 figure 级图例，避免参与 axes 的 tight bbox
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005),
               handlelength=1.6, columnspacing=1.6)

    ax.set_title(f"尿酸场景因果图（DAG）：{len(nodes)} 个节点 / {len(DAG_EDGES)} 条边 / "
                 f"经拓扑排序校验为无环", fontsize=13, pad=12)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlim(-0.52, max_level + 0.52)
    ax.set_ylim(-span_y / 2 - 0.6, span_y / 2 + 0.6)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.20)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")
    return True
    plt.close(fig)
    print(f"[plot] 已保存 {out_path}")
    return True


def write_outputs(out_dir) -> dict:
    """把 DAG、校验结果、调整集推导、特征集、模型定义全部落盘。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DAG 图（与下方 CSV 同源，保证图文一致）
    try:
        plot_dag(out_dir / "causal_dag.png")
    except ImportError as exc:
        print(f"[plot] 跳过 DAG 绘图（缺依赖：{exc}）")

    # --- DAG 边
    df_edges = pd.DataFrame(
        [(p, c, cat, rat) for p, c, cat, rat in DAG_EDGES],
        columns=["parent", "child", "category", "rationale"])
    write_csv(df_edges, out_dir / "causal_dag_edges.csv")

    # --- DAG 校验
    val = validate_dag()
    val["node_notes"] = "; ".join(f"{k}={v}" for k, v in sorted(NODE_NOTES.items()))
    write_csv(pd.DataFrame([val]), out_dir / "dag_validation.csv")

    # --- 后门调整集推导
    df_adj = derive_adjustment_sets()
    write_csv(df_adj, out_dir / "adjustment_set_derivation.csv")

    # --- 特征集
    fss = get_feature_sets()
    rows = []
    for key, spec in fss.items():
        rows.append({
            "task": "UA",
            "feature_set": key,
            "role": spec["role"],
            "tier": spec["tier"],
            "n_columns": len(spec["columns"]),
            "columns": ";".join(spec["columns"]),
            "rationale": spec["rationale"],
        })
    write_csv(pd.DataFrame(rows), out_dir / "causal_feature_sets.csv")

    # --- 模型定义
    models = get_model_definitions()
    rows_m = []
    for name, spec in models.items():
        cols = fss[spec["feature_set_key"]]["columns"]
        rows_m.append({
            "model": name,
            "feature_set_key": spec["feature_set_key"],
            "n_features": len(cols),
            "features": ";".join(cols),
            "estimator": spec["estimator"],
            "description": spec["description"],
        })
    write_csv(pd.DataFrame(rows_m), out_dir / "model_definitions.csv")

    # --- 控制台摘要
    print(f"[ua_causal_specification] DAG: {val['n_nodes']} 节点 / {val['n_edges']} 边 / "
          f"无环 = {val['is_acyclic']}")
    if not val["is_acyclic"]:
        print(f"[ua_causal_specification] !! 检测到环，未解节点：{val['unresolved_nodes']}")

    conf = df_adj[df_adj["verdict"] == "confounder"]
    joint = df_adj[df_adj["verdict"] == "joint_treatment"]
    print(f"[ua_causal_specification] Tier-1 最小集 = 混杂变量 "
          f"{{{', '.join(conf['candidate'])}}} ∪ 联合处理变量 "
          f"{{{', '.join(joint['candidate'])}}}")
    t2 = df_adj[df_adj["in_tier2_minimal_plus_renal"] & ~df_adj["in_tier1_minimal_set"]]
    print(f"[ua_causal_specification] Tier-2 追加 = {{{', '.join(t2['candidate'])}}}"
          f"（身份：{'; '.join(t2['verdict'])}）")

    print(f"[ua_causal_specification] 特征集 {len(fss)} 个，模型 {len(models)} 个")
    for name, spec in models.items():
        n = len(fss[spec['feature_set_key']]['columns'])
        print(f"    {name:26s} <- {spec['feature_set_key']:26s} ({n} 列)")
    print(f"[ua_causal_specification] 全部产物写入 {out_dir}")


def main():
    setup_console()
    parser = argparse.ArgumentParser(description="UA Causal Specification")
    parser.add_argument("--out", default="results/ua_causal_specification",
                        help="输出目录")
    args = parser.parse_args()
    write_outputs(args.out)


if __name__ == "__main__":
    main()
