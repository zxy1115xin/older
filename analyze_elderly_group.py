import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyze_gait_features import extract_subject_features
from read_xlsx import (
    get_right_side_sheets,
    load_curve_long,
    load_demo_info,
    sheet_name_to_chinese_title,
)


FEATURE_COLUMNS = [
    "PeakValue",
    "PeakCycle",
    "TroughValue",
    "TroughCycle",
    "ROM",
    "MeanValue",
    "StdValue",
    "AUC",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="比较老年组与非老年组的步态特征差异。")
    parser.add_argument(
        "--elderly-age",
        type=float,
        default=60.0,
        help="老年组年龄阈值，默认 60 岁。",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="输出和绘图展示的显著特征数量。",
    )
    return parser.parse_args()


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return np.nan
    x_std = np.std(x, ddof=1)
    y_std = np.std(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * x_std**2 + (len(y) - 1) * y_std**2) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m

    for rank, idx in enumerate(order):
        adjusted_sorted[rank] = (m - rank) * p_values[idx]

    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(adjusted_sorted[rank], 1.0)
    return adjusted


def prepare_subject_feature_table(demo_path: str, gait_path: str) -> pd.DataFrame:
    demo = load_demo_info(demo_path)
    right_sheets = get_right_side_sheets(gait_path)
    rows = []

    for sheet_name in right_sheets:
        curve_long = load_curve_long(gait_path, sheet_name=sheet_name, value_col="Value")
        merged = curve_long.merge(demo, on="Subject", how="inner")
        merged = merged.dropna(subset=["Age"])
        if merged.empty:
            continue

        title_cn = sheet_name_to_chinese_title(sheet_name)
        feature_df = extract_subject_features(merged)
        if feature_df.empty:
            continue

        feature_df.insert(0, "TitleCN", title_cn)
        feature_df.insert(0, "SheetName", sheet_name)
        rows.append(feature_df)

    if not rows:
        raise ValueError("未生成任何受试者特征，无法进行老年组分析。")

    return pd.concat(rows, ignore_index=True)


def compare_elderly_groups(subject_feature_df: pd.DataFrame, elderly_age: float) -> pd.DataFrame:
    result_rows = []
    work_df = subject_feature_df.copy()
    work_df["ElderlyGroup"] = np.where(work_df["Age"] >= elderly_age, "Elderly", "NonElderly")

    for (sheet_name, title_cn), sheet_df in work_df.groupby(["SheetName", "TitleCN"], observed=True):
        for feature in FEATURE_COLUMNS:
            elderly_vals = sheet_df.loc[sheet_df["ElderlyGroup"] == "Elderly", feature].dropna().to_numpy()
            non_vals = sheet_df.loc[sheet_df["ElderlyGroup"] == "NonElderly", feature].dropna().to_numpy()
            if len(elderly_vals) < 2 or len(non_vals) < 2:
                continue

            mann_u, mann_p = stats.mannwhitneyu(elderly_vals, non_vals, alternative="two-sided")
            t_stat, t_p = stats.ttest_ind(elderly_vals, non_vals, equal_var=False, nan_policy="omit")
            effect = cohens_d(elderly_vals, non_vals)

            result_rows.append(
                {
                    "SheetName": sheet_name,
                    "TitleCN": title_cn,
                    "Feature": feature,
                    "ElderlyThreshold": elderly_age,
                    "ElderlyN": int(len(elderly_vals)),
                    "NonElderlyN": int(len(non_vals)),
                    "ElderlyMean": float(np.mean(elderly_vals)),
                    "NonElderlyMean": float(np.mean(non_vals)),
                    "MeanDiff_ElderlyMinusNon": float(np.mean(elderly_vals) - np.mean(non_vals)),
                    "ElderlyMedian": float(np.median(elderly_vals)),
                    "NonElderlyMedian": float(np.median(non_vals)),
                    "MannWhitneyU": float(mann_u),
                    "MannWhitneyP": float(mann_p),
                    "TTestStat": float(t_stat),
                    "TTestP": float(t_p),
                    "CohensD": effect,
                    "AbsCohensD": float(abs(effect)) if pd.notna(effect) else np.nan,
                }
            )

    result_df = pd.DataFrame(result_rows)
    if result_df.empty:
        return result_df

    adjusted = holm_adjust(result_df["MannWhitneyP"].fillna(1.0).astype(float).tolist())
    result_df["MannWhitneyP_Holm"] = adjusted
    result_df["Significant_p<0.05"] = result_df["MannWhitneyP_Holm"] < 0.05
    return result_df.sort_values(
        ["Significant_p<0.05", "MannWhitneyP_Holm", "AbsCohensD"],
        ascending=[False, True, False],
    ).reset_index(drop=True)


def plot_top_boxplots(
    subject_feature_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    elderly_age: float,
    output_png: Path,
    top_n: int,
) -> None:
    top_df = comparison_df.head(top_n).copy()
    if top_df.empty:
        return

    plot_df = subject_feature_df.copy()
    plot_df["ElderlyGroup"] = np.where(plot_df["Age"] >= elderly_age, "Elderly", "NonElderly")

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    cols = 2
    rows = int(np.ceil(len(top_df) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(4.8 * rows, 5.5)), dpi=140)
    axes = np.atleast_1d(axes).flatten()
    colors = {"NonElderly": "#93c5fd", "Elderly": "#fca5a5"}

    for ax, row in zip(axes, top_df.itertuples(index=False)):
        subset = plot_df[plot_df["SheetName"] == row.SheetName]
        values = [
            subset.loc[subset["ElderlyGroup"] == "NonElderly", row.Feature].dropna().to_numpy(),
            subset.loc[subset["ElderlyGroup"] == "Elderly", row.Feature].dropna().to_numpy(),
        ]
        bp = ax.boxplot(
            values,
            labels=["NonElderly", "Elderly"],
            patch_artist=True,
            medianprops={"color": "#111827", "linewidth": 1.6},
            boxprops={"linewidth": 1.2},
            whiskerprops={"linewidth": 1.1},
            capprops={"linewidth": 1.1},
        )
        for patch, label in zip(bp["boxes"], ["NonElderly", "Elderly"]):
            patch.set_facecolor(colors[label])
            patch.set_alpha(0.8)

        for idx, group_name in enumerate(["NonElderly", "Elderly"], start=1):
            y = subset.loc[subset["ElderlyGroup"] == group_name, row.Feature].dropna().to_numpy()
            if len(y) == 0:
                continue
            x = np.random.normal(loc=idx, scale=0.05, size=len(y))
            ax.scatter(x, y, s=16, color="#334155", alpha=0.55)

        ax.set_title(
            f"{row.TitleCN}\n{row.Feature} | Holm p={row.MannWhitneyP_Holm:.3g} | d={row.CohensD:.2f}",
            fontsize=10.5,
            fontweight="bold",
        )
        ax.set_ylabel(row.Feature)

    for ax in axes[len(top_df) :]:
        ax.axis("off")

    fig.suptitle(
        f"老年组(>={elderly_age:.0f}岁) vs 非老年组的关键步态特征箱线图",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    demo_path = "data/01_Demo_PhysEx.xlsx"
    gait_path = "data/02_Overview_comf.xlsx"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    subject_feature_df = prepare_subject_feature_table(demo_path, gait_path)
    comparison_df = compare_elderly_groups(subject_feature_df, elderly_age=args.elderly_age)
    if comparison_df.empty:
        raise ValueError("老年组与非老年组的可比较特征为空。")

    output_excel = output_dir / "elderly_group_analysis.xlsx"
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        subject_feature_df.to_excel(writer, sheet_name="subject_features", index=False)
        comparison_df.to_excel(writer, sheet_name="elderly_vs_nonelderly", index=False)
        comparison_df.head(args.top_n).to_excel(writer, sheet_name="top_features", index=False)

    output_png = output_dir / "elderly_group_top_boxplots.png"
    plot_top_boxplots(
        subject_feature_df=subject_feature_df,
        comparison_df=comparison_df,
        elderly_age=args.elderly_age,
        output_png=output_png,
        top_n=args.top_n,
    )

    print(f"已输出老年组比较结果: {output_excel}")
    print(f"已输出箱线图: {output_png}")
    print("显著性最强的前10个特征:")
    print(
        comparison_df[
            ["SheetName", "Feature", "ElderlyMean", "NonElderlyMean", "MannWhitneyP_Holm", "CohensD"]
        ]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
