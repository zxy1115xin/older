from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from read_xlsx import (
    get_right_side_sheets,
    load_curve_long,
    load_demo_info,
    sheet_name_to_chinese_title,
)


def extract_subject_features(merged: pd.DataFrame) -> pd.DataFrame:
    """提取每位受试者在单个指标下的关键步态特征。"""
    rows = []
    for (subject, age, age_group), group_df in merged.groupby(
        ["Subject", "Age", "AgeGroup"], observed=True
    ):
        ordered = group_df.sort_values("Cycle").reset_index(drop=True)
        values = ordered["Value"].to_numpy(dtype=float)
        cycles = ordered["Cycle"].to_numpy(dtype=float)

        if len(values) == 0:
            continue

        peak_idx = int(np.nanargmax(values))
        trough_idx = int(np.nanargmin(values))

        rows.append(
            {
                "Subject": subject,
                "Age": float(age),
                "AgeGroup": str(age_group),
                "PeakValue": float(values[peak_idx]),
                "PeakCycle": float(cycles[peak_idx]),
                "TroughValue": float(values[trough_idx]),
                "TroughCycle": float(cycles[trough_idx]),
                "ROM": float(values[peak_idx] - values[trough_idx]),
                "MeanValue": float(np.nanmean(values)),
                "StdValue": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                "AUC": float(np.trapz(values, cycles)),
            }
        )

    return pd.DataFrame(rows)


def analyze_feature_age_relationship(
    feature_df: pd.DataFrame, sheet_name: str, title_cn: str
) -> pd.DataFrame:
    """用连续年龄评估每个步态特征的变化趋势。"""
    feature_cols = [
        "PeakValue",
        "PeakCycle",
        "TroughValue",
        "TroughCycle",
        "ROM",
        "MeanValue",
        "StdValue",
        "AUC",
    ]
    rows = []

    for feature in feature_cols:
        valid = feature_df[["Age", feature]].dropna()
        if len(valid) < 3:
            continue

        spearman = stats.spearmanr(valid["Age"], valid[feature])
        pearson = stats.pearsonr(valid["Age"], valid[feature])
        slope, intercept, r_value, p_value, stderr = stats.linregress(
            valid["Age"], valid[feature]
        )

        rows.append(
            {
                "SheetName": sheet_name,
                "TitleCN": title_cn,
                "Feature": feature,
                "SampleSize": int(len(valid)),
                "SpearmanRho": float(spearman.statistic),
                "SpearmanP": float(spearman.pvalue),
                "PearsonR": float(pearson.statistic),
                "PearsonP": float(pearson.pvalue),
                "LinearSlopePerYear": float(slope),
                "LinearIntercept": float(intercept),
                "LinearR": float(r_value),
                "LinearP": float(p_value),
                "LinearStdErr": float(stderr),
                "AbsSpearmanRho": float(abs(spearman.statistic)),
            }
        )

    return pd.DataFrame(rows)


def summarize_feature_by_age_group(
    feature_df: pd.DataFrame, sheet_name: str, title_cn: str
) -> pd.DataFrame:
    """按年龄段汇总特征均值，便于查看老化方向。"""
    metrics = [
        "PeakValue",
        "PeakCycle",
        "TroughValue",
        "TroughCycle",
        "ROM",
        "MeanValue",
        "StdValue",
        "AUC",
    ]

    summary = (
        feature_df.groupby("AgeGroup", observed=True)[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        col if isinstance(col, str) else "_".join([str(x) for x in col if str(x)])
        for col in summary.columns.to_flat_index()
    ]
    summary.insert(0, "TitleCN", title_cn)
    summary.insert(0, "SheetName", sheet_name)
    return summary


def plot_feature_correlation_heatmap(corr_df: pd.DataFrame, output_png: Path) -> None:
    """绘制各指标特征与年龄相关性的热图。"""
    if corr_df.empty:
        return

    pivot = corr_df.pivot(index="TitleCN", columns="Feature", values="SpearmanRho")
    pivot = pivot.sort_index()

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig_h = max(6, 0.42 * len(pivot.index))
    fig, ax = plt.subplots(figsize=(12, fig_h), dpi=140)
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("各步态指标特征与年龄的 Spearman 相关性热图", fontsize=15, fontweight="bold")

    for row_idx in range(len(pivot.index)):
        for col_idx in range(len(pivot.columns)):
            value = pivot.iat[row_idx, col_idx]
            if pd.isna(value):
                label = ""
            else:
                label = f"{value:.2f}"
            ax.text(col_idx, row_idx, label, ha="center", va="center", fontsize=8, color="#111827")

    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Spearman rho", rotation=90)

    fig.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    demo_path = "data/01_Demo_PhysEx.xlsx"
    gait_path = "data/02_Overview_comf.xlsx"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    demo = load_demo_info(demo_path)
    right_sheets = get_right_side_sheets(gait_path)
    if not right_sheets:
        raise ValueError("未识别到可分析的右侧步态 sheet。")

    all_subject_features = []
    all_correlations = []
    all_group_summaries = []

    for sheet_name in right_sheets:
        curve_long = load_curve_long(gait_path, sheet_name=sheet_name, value_col="Value")
        merged = curve_long.merge(demo, on="Subject", how="inner")
        merged = merged.dropna(subset=["AgeGroup"])
        if merged.empty:
            continue

        title_cn = sheet_name_to_chinese_title(sheet_name)
        feature_df = extract_subject_features(merged)
        if feature_df.empty:
            continue

        feature_df.insert(0, "TitleCN", title_cn)
        feature_df.insert(0, "SheetName", sheet_name)

        corr_df = analyze_feature_age_relationship(feature_df, sheet_name, title_cn)
        group_summary_df = summarize_feature_by_age_group(feature_df, sheet_name, title_cn)

        all_subject_features.append(feature_df)
        all_correlations.append(corr_df)
        all_group_summaries.append(group_summary_df)

    if not all_subject_features:
        raise ValueError("未生成任何可用的受试者特征结果。")

    subject_feature_df = pd.concat(all_subject_features, ignore_index=True)
    corr_df = pd.concat(all_correlations, ignore_index=True)
    group_summary_df = pd.concat(all_group_summaries, ignore_index=True)

    top_corr_df = corr_df.sort_values(
        ["AbsSpearmanRho", "SpearmanP"], ascending=[False, True]
    ).reset_index(drop=True)

    output_excel = output_dir / "gait_feature_deep_analysis.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        subject_feature_df.to_excel(writer, sheet_name="subject_features", index=False)
        corr_df.to_excel(writer, sheet_name="age_correlations", index=False)
        top_corr_df.to_excel(writer, sheet_name="top_age_related_features", index=False)
        group_summary_df.to_excel(writer, sheet_name="age_group_summary", index=False)

    heatmap_png = output_dir / "gait_feature_age_correlation_heatmap.png"
    plot_feature_correlation_heatmap(corr_df, heatmap_png)

    print(f"已输出深度分析结果: {output_excel}")
    print(f"已输出相关性热图: {heatmap_png}")
    print("年龄相关性最强的前10个特征:")
    print(
        top_corr_df[
            ["SheetName", "Feature", "SpearmanRho", "SpearmanP", "LinearSlopePerYear"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
