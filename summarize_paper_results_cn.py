import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将步态分析结果自动整理成中文论文摘要。")
    parser.add_argument(
        "--top-n",
        type=int,
        default=8,
        help="每个分析模块纳入摘要的前 N 个结果。",
    )
    return parser.parse_args()


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "p=NA"
    if p_value < 0.001:
        return "p<0.001"
    return f"p={p_value:.3f}"


def format_signed(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{value:+.{digits}f}"


def load_tables(output_dir: Path) -> tuple[pd.DataFrame, ...]:
    deep_path = output_dir / "gait_feature_deep_analysis.xlsx"
    elderly_path = output_dir / "elderly_group_analysis.xlsx"
    model_path = output_dir / "age_prediction_model_results.xlsx"

    missing = [str(path) for path in [deep_path, elderly_path, model_path] if not path.exists()]
    if missing:
        raise FileNotFoundError("缺少以下结果文件，请先运行前面的分析脚本：\n" + "\n".join(missing))

    subject_features = pd.read_excel(deep_path, sheet_name="subject_features")
    age_corr = pd.read_excel(deep_path, sheet_name="top_age_related_features")
    elderly_cmp = pd.read_excel(elderly_path, sheet_name="elderly_vs_nonelderly")
    regression_summary = pd.read_excel(model_path, sheet_name="age_regression_summary")
    classification_summary = pd.read_excel(model_path, sheet_name="elderly_cls_summary")
    feature_importance = pd.read_excel(model_path, sheet_name="feature_importance")
    return (
        subject_features,
        age_corr,
        elderly_cmp,
        regression_summary,
        classification_summary,
        feature_importance,
    )


def build_dataset_section(subject_features: pd.DataFrame) -> str:
    subject_df = subject_features[["Subject", "Age", "AgeGroup"]].drop_duplicates()
    n_subjects = len(subject_df)
    age_mean = subject_df["Age"].mean()
    age_std = subject_df["Age"].std()
    age_min = subject_df["Age"].min()
    age_max = subject_df["Age"].max()
    age_group_counts = (
        subject_df["AgeGroup"].astype(str).value_counts().sort_index().to_dict()
    )
    age_group_text = "，".join([f"{k}: {v}例" for k, v in age_group_counts.items()])

    return (
        f"本研究共纳入 {n_subjects} 名受试者，年龄范围 {age_min:.0f}-{age_max:.0f} 岁，"
        f"平均年龄为 {age_mean:.2f}±{age_std:.2f} 岁。"
        f"按年龄段划分后，各组样本量分别为 {age_group_text}。"
    )


def build_age_trend_section(age_corr: pd.DataFrame, top_n: int) -> str:
    top_df = age_corr.head(top_n).copy()
    lines = []
    for row in top_df.itertuples(index=False):
        direction = "增加" if row.SpearmanRho > 0 else "降低"
        lines.append(
            f"{row.TitleCN} 的 {row.Feature} 随年龄{direction}"
            f"（rho={row.SpearmanRho:.3f}，{format_p_value(row.SpearmanP)}，"
            f"线性斜率={format_signed(row.LinearSlopePerYear, 3)}/年）"
        )

    return (
        "连续年龄趋势分析显示，多项步态特征与年龄存在显著相关。"
        "相关性最强的指标主要集中在髋关节冠状面活动、髋屈伸功率以及踝关节屈伸角度等方面。"
        "其中，" + "；".join(lines) + "。"
    )


def build_elderly_section(elderly_cmp: pd.DataFrame, top_n: int) -> str:
    top_df = elderly_cmp.head(top_n).copy()
    elderly_threshold = float(top_df["ElderlyThreshold"].iloc[0])
    elderly_n = int(top_df["ElderlyN"].iloc[0])
    nonelderly_n = int(top_df["NonElderlyN"].iloc[0])

    lines = []
    for row in top_df.itertuples(index=False):
        direction = "高于" if row.MeanDiff_ElderlyMinusNon > 0 else "低于"
        lines.append(
            f"老年组在 {row.TitleCN} 的 {row.Feature} {direction}非老年组"
            f"（均值差={format_signed(row.MeanDiff_ElderlyMinusNon, 3)}，"
            f"{format_p_value(row.MannWhitneyP_Holm)}，Cohen's d={row.CohensD:.2f}）"
        )

    return (
        f"以 {elderly_threshold:.0f} 岁为阈值进行分组后，老年组（n={elderly_n}）与非老年组（n={nonelderly_n}）"
        "在多个关键步态特征上表现出显著差异。"
        "差异最明显的结果包括：" + "；".join(lines) + "。"
    )


def build_model_section(
    regression_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_n: int,
) -> str:
    reg_best = regression_summary.sort_values(["R2", "MAE"], ascending=[False, True]).iloc[0]
    cls_best = classification_summary.sort_values(
        ["ROC_AUC", "Accuracy"], ascending=[False, False]
    ).iloc[0]

    pure_gait_importance = feature_importance[
        ~feature_importance["FeatureName"].astype(str).str.startswith("AgeGroup_")
    ].head(top_n)
    imp_text = "；".join(
        [
            f"{row.FeatureName}（重要性={row.Importance:.3f}）"
            for row in pure_gait_importance.itertuples(index=False)
        ]
    )

    caution = (
        "需要注意的是，模型结果表中的 AgeGroup 属于由年龄直接派生的变量，"
        "因此不宜作为独立生物力学预测因子进行解释。"
    )

    return (
        f"在预测建模方面，{reg_best.Model} 在年龄回归任务中表现最佳"
        f"（MAE={reg_best.MAE:.3f} 岁，R²={reg_best.R2:.3f}）；"
        f"{cls_best.Model} 在老年判别任务中表现最佳"
        f"（Accuracy={cls_best.Accuracy:.3f}，ROC_AUC={cls_best.ROC_AUC:.3f}）。"
        f"在剔除 AgeGroup 后，最重要的步态特征主要包括：{imp_text}。"
        + caution
    )


def build_conclusion_section(
    age_corr: pd.DataFrame, elderly_cmp: pd.DataFrame, feature_importance: pd.DataFrame
) -> str:
    top_age_titles = age_corr.head(5)["TitleCN"].dropna().tolist()
    top_elderly_titles = elderly_cmp.head(5)["TitleCN"].dropna().tolist()
    top_model_features = (
        feature_importance[
            ~feature_importance["FeatureName"].astype(str).str.startswith("AgeGroup_")
        ]["FeatureName"]
        .head(5)
        .tolist()
    )

    repeated_titles = list(dict.fromkeys(top_age_titles + top_elderly_titles))
    repeated_text = "、".join(repeated_titles[:6]) if repeated_titles else "髋、膝、踝多关节指标"
    model_text = "、".join(top_model_features[:5]) if top_model_features else "多项步态特征"

    return (
        "综合连续年龄分析、老年组比较以及预测建模结果可以认为，"
        f"{repeated_text} 是反映衰老相关步态改变的核心观察窗口。"
        "总体上，老化过程不仅表现为关节活动幅度下降，还伴随着部分功率输出和关键事件时相的改变，"
        "提示老年人在步态控制中可能存在活动范围收缩、远端控制减弱及近端代偿增强等模式。"
        f"其中，{model_text} 具有较高的区分和预测价值，可作为后续老年步态评估、风险筛查和机制研究的候选特征。"
    )


def build_limitations_section() -> str:
    return (
        "本摘要基于当前脚本自动提取的统计结果生成，主要反映相关性和组间差异，"
        "尚不能直接推断因果关系。后续研究可进一步补充性别、身高体重、步速控制、"
        "共病状态等混杂因素校正，并在独立样本中验证模型稳定性。"
    )


def build_markdown_report(
    subject_features: pd.DataFrame,
    age_corr: pd.DataFrame,
    elderly_cmp: pd.DataFrame,
    regression_summary: pd.DataFrame,
    classification_summary: pd.DataFrame,
    feature_importance: pd.DataFrame,
    top_n: int,
) -> str:
    sections = [
        "# 步态衰老研究结果摘要",
        "## 样本概况",
        build_dataset_section(subject_features),
        "## 年龄趋势结果",
        build_age_trend_section(age_corr, top_n),
        "## 老年组比较结果",
        build_elderly_section(elderly_cmp, top_n),
        "## 预测建模结果",
        build_model_section(
            regression_summary, classification_summary, feature_importance, top_n
        ),
        "## 综合结论",
        build_conclusion_section(age_corr, elderly_cmp, feature_importance),
        "## 研究提示",
        build_limitations_section(),
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path("output")
    (
        subject_features,
        age_corr,
        elderly_cmp,
        regression_summary,
        classification_summary,
        feature_importance,
    ) = load_tables(output_dir)

    report = build_markdown_report(
        subject_features=subject_features,
        age_corr=age_corr,
        elderly_cmp=elderly_cmp,
        regression_summary=regression_summary,
        classification_summary=classification_summary,
        feature_importance=feature_importance,
        top_n=args.top_n,
    )

    md_path = output_dir / "paper_result_summary_cn.md"
    txt_path = output_dir / "paper_result_summary_cn.txt"
    md_path.write_text(report, encoding="utf-8")
    txt_path.write_text(report, encoding="utf-8")

    print(f"已输出中文论文摘要: {md_path}")
    print(f"已输出纯文本版本: {txt_path}")
    preview = report.encode("gbk", errors="replace").decode("gbk", errors="replace")
    print("\n摘要预览：\n")
    print(preview)


if __name__ == "__main__":
    main()
