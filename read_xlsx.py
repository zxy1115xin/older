from pathlib import Path
from itertools import combinations
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy import stats


def load_demo_info(file_path: str) -> pd.DataFrame:
    """读取样本基础信息，提取 Subject、Age 和年龄段。"""
    # 生成受试者级别的基础信息表，后续所有步态数据都会按 Subject 与它做合并。
    # 该表前 3 行为说明，真实表头在第 4 行（header=3）
    demo = pd.read_excel(file_path, header=3, engine="openpyxl")
    demo = demo.rename(columns={
        "Subject": "Subject",
        "Age\n (yrs)": "Age",
    })

    # 第一列通常是空列，按名称模式清理
    drop_cols = [c for c in demo.columns if str(c).startswith("Unnamed")]
    demo = demo.drop(columns=drop_cols, errors="ignore")

    demo["Subject"] = demo["Subject"].astype(str).str.strip()
    demo["Age"] = pd.to_numeric(demo["Age"], errors="coerce")

    # 年龄段标签既用于统计分析，也用于绘图分组展示。
    bins = [18, 30, 40, 50, 60, 70, 200]
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    demo["AgeGroup"] = pd.cut(demo["Age"], bins=bins, labels=labels, right=False)
    return demo[["Subject", "Age", "AgeGroup"]].dropna(subset=["Subject", "Age"])


def get_right_side_sheets(file_path: str) -> list[str]:
    """读取工作簿中所有右侧肢体相关 sheet 名。"""
    xls = pd.ExcelFile(file_path, engine="openpyxl")
    # 只筛选右侧下肢相关 sheet，避免无关表进入同一批处理流程。
    # 规则说明：
    # 1) GRF_R_*      -> 右侧地反力
    # 2) Rotation_R*  -> 右侧关节角度
    # 3) Moment_R*    -> 右侧关节力矩
    # 4) Power_R*     -> 右侧关节功率
    # 5) RProgression -> 右侧进程角
    pattern = re.compile(r"^(GRF_R_|(?:Rotation|Moment|Power)_R|RProgression)", re.IGNORECASE)
    right_sheets = [s for s in xls.sheet_names if pattern.search(str(s))]
    return sorted(right_sheets)


def load_curve_long(file_path: str, sheet_name: str, value_col: str = "Value") -> pd.DataFrame:
    """读取任意曲线 sheet，统一转换成长表结构。"""
    curve_df = pd.read_excel(file_path, sheet_name=sheet_name, engine="openpyxl")

    subject_col = "Gait Cycle \n(1-100%)"
    if subject_col not in curve_df.columns:
        raise KeyError(f"未找到样本编号列: {subject_col}")

    curve_df = curve_df.rename(columns={subject_col: "Subject"})
    curve_df["Subject"] = curve_df["Subject"].astype(str).str.strip()

    # 原始表是宽表结构，每个步态周期点占一列；转成长表后更适合 groupby 和绘图。
    # 第 1 列是说明文本，后续 100 列是 gait cycle 1..100
    curve_cols = [c for c in curve_df.columns if str(c).replace(".0", "").isdigit()]
    if not curve_cols:
        raise KeyError("未识别到步态周期列（1..100）")

    long_df = curve_df[["Subject", *curve_cols]].melt(
        id_vars="Subject", var_name="Cycle", value_name=value_col
    )
    long_df["Cycle"] = pd.to_numeric(long_df["Cycle"], errors="coerce")
    long_df[value_col] = pd.to_numeric(long_df[value_col], errors="coerce")
    return long_df.dropna(subset=["Subject", "Cycle", value_col])


def p_to_stars(p_value: float) -> str:
    """将 p 值转换为显著性星号。"""
    if pd.isna(p_value):
        return "ns"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def sheet_name_to_chinese_title(sheet_name: str) -> str:
    """将类似 Moment_RHipAbAd_comf 的 sheet 名转换为中文标题。"""
    speed_map = {
        "comf": "舒适速度",
        "fast": "快速",
        "slow": "慢速",
    }
    type_map = {
        "Rotation": "角度",
        "Moment": "力矩",
        "Power": "功率",
    }
    side_map = {"R": "右侧", "L": "左侧"}
    component_map = {
        "TrunkTilt": "躯干前后倾",
        "TrunkFlex": "躯干侧屈",
        "TrunkRot": "躯干旋转",
        "PelvicTil": "骨盆前后倾",
        "PelvicObl": "骨盆倾斜",
        "PelvRot": "骨盆旋转",
        "HipFlex": "髋关节屈伸",
        "HipAbAd": "髋关节内收/外展",
        "HipRot": "髋关节旋转",
        "KneeFlex": "膝关节屈伸",
        "KneeAbAd": "膝关节内收/外展",
        "KneeRot": "膝关节旋转",
        "AnkleFlex": "踝关节屈伸",
        "AnklePron": "踝关节旋前/旋后",
        "Progression": "足进展角",
    }

    raw = str(sheet_name).strip()
    parts = raw.split("_")

    speed_cn = speed_map.get(parts[-1].lower(), "该速度") if parts else "该速度"

    # 处理 GRF_R_AP_comf / GRF_R_vert_comf
    if len(parts) >= 4 and parts[0].upper() == "GRF":
        side_cn = side_map.get(parts[1].upper(), "")
        dir_map = {
            "AP": "前后向",
            "VERT": "垂直向",
            "ML": "内外侧",
        }
        direction_cn = dir_map.get(parts[2].upper(), parts[2])
        return f"{side_cn}{direction_cn}地面反作用力（{speed_cn}）"

    # 处理 Rotation_RKneeFlex_comf / Moment_RHipAbAd_comf / Power_RAnkleFlex_comf
    if len(parts) >= 3 and parts[0] in type_map:
        signal_type = parts[0]
        side_and_component = parts[1]
        speed_cn = speed_map.get(parts[2].lower(), speed_cn)

        side_cn = ""
        comp_key = side_and_component
        if side_and_component and side_and_component[0].upper() in side_map:
            side_cn = side_map[side_and_component[0].upper()]
            comp_key = side_and_component[1:]

        comp_cn = component_map.get(comp_key, comp_key)
        type_cn = type_map[signal_type]
        return f"{side_cn}{comp_cn}{type_cn}（{speed_cn}）"

    if raw.startswith("RProgression"):
        return f"右侧足进展角（{speed_cn}）"

    # 无法识别时保留原名称，避免因命名异常中断
    return f"{raw}（{speed_cn}）"


def plot_power_by_age_group(
    merged_long: pd.DataFrame,
    output_png: str,
    group_n: pd.Series,
    signif_pairwise_df: pd.DataFrame,
    sheet_name: str,
    value_col: str = "Value",
) -> None:
    """按年龄段绘制曲线 + 统计柱状图（峰值/谷值），并附带标准差带。"""
    # 左图保留完整步态周期趋势，右图提取峰值/谷值，方便直接做组间特征比较。
    summary = (
        merged_long.groupby(["AgeGroup", "Cycle"], observed=True)[value_col]
        .agg(["mean", "std"])
        .reset_index()
    )
    subject_feature = (
        merged_long.groupby(["AgeGroup", "Subject"], observed=True)[value_col]
        .agg(PeakValue="max", TroughValue="min")
        .reset_index()
    )
    feature_summary = (
        subject_feature.groupby("AgeGroup", observed=True)[["PeakValue", "TroughValue"]]
        .agg(["mean", "std"])
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    # 演示风：提高整体对比度，投影时更清晰
    plt.rcParams["figure.facecolor"] = "#ffffff"
    plt.rcParams["axes.facecolor"] = "#f8fafc"

    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.2, 6.1),
        dpi=130,
        gridspec_kw={"width_ratios": [2.9, 1.7], "wspace": 0.16},
    )
    title_cn = sheet_name_to_chinese_title(sheet_name)
    fig.suptitle(
        f"{title_cn} 年龄组对比",
        fontsize=20,
        fontweight="bold",
        y=0.995,
        color="#0b1220",
    )
    age_groups_for_plot = [g for g, _ in summary.groupby("AgeGroup", observed=True)]
    # 左图使用连续渐变色，避免离散高饱和配色造成突兀感
    cmap = plt.get_cmap("YlGnBu")
    color_positions = np.linspace(0.35, 0.9, max(len(age_groups_for_plot), 1))
    gradient_colors = [cmap(pos) for pos in color_positions]
    # 右图柱状图也从同一渐变色中取色，保证左右图视觉风格一致
    peak_bar_color = cmap(0.78)
    trough_bar_color = cmap(0.52)

    for i, (age_group, group_df) in enumerate(summary.groupby("AgeGroup", observed=True)):
        color = gradient_colors[i]
        n = int(group_n.get(age_group, 0))
        x = group_df["Cycle"].to_numpy()
        y = group_df["mean"].to_numpy()
        s = group_df["std"].fillna(0).to_numpy()

        ax.plot(x, y, lw=2.9, color=color, label=f"{age_group}岁 (n={n})")
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, linewidth=0)

    ax.set_title("步态周期曲线（均值 ± 标准差）", fontsize=16, fontweight="bold", pad=10, color="#111827")
    ax.set_xlabel("步态周期（%）", fontsize=13)
    ax.set_ylabel(value_col, fontsize=13)
    ax.set_xlim(1, 100)

    y_min = float((summary["mean"] - summary["std"].fillna(0)).min())
    y_max = float((summary["mean"] + summary["std"].fillna(0)).max())
    y_pad = 0.08 * (y_max - y_min)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.axhline(0, color="#334155", lw=1.2, alpha=0.95)
    ax.grid(which="major", axis="both", linestyle="--", color="#94a3b8", linewidth=0.8, alpha=0.9)
    ax.grid(which="minor", axis="both", linestyle=":", color="#cbd5e1", linewidth=0.55, alpha=0.95)

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.tick_params(axis="both", which="major", direction="in", length=6, width=1.1, labelsize=11)
    ax.tick_params(axis="both", which="minor", direction="in", length=3.5, width=0.8)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(1.2)
        ax.spines[spine].set_color("#334155")

    legend = ax.legend(
        title="年龄组",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#475569",
        framealpha=1.0,
        fontsize=10.5,
        title_fontsize=12,
    )
    legend.get_frame().set_linewidth(1.0)

    age_order = [g for g in group_n.index if g in feature_summary.index]
    x_pos = np.arange(len(age_order))
    peak_mean = feature_summary["PeakValue"]["mean"].reindex(age_order).to_numpy()
    peak_std = feature_summary["PeakValue"]["std"].reindex(age_order).fillna(0).to_numpy()
    trough_mean = feature_summary["TroughValue"]["mean"].reindex(age_order).to_numpy()
    trough_std = feature_summary["TroughValue"]["std"].reindex(age_order).fillna(0).to_numpy()

    bar_w = 0.38
    ax2.bar(
        x_pos - bar_w / 2,
        peak_mean,
        width=bar_w,
        yerr=peak_std,
        capsize=4,
        color=peak_bar_color,
        alpha=0.95,
        edgecolor="#0f172a",
        linewidth=0.9,
        label="峰值",
    )
    ax2.bar(
        x_pos + bar_w / 2,
        trough_mean,
        width=bar_w,
        yerr=trough_std,
        capsize=4,
        color=trough_bar_color,
        alpha=0.95,
        edgecolor="#0f172a",
        linewidth=0.9,
        label="谷值",
    )

    ax2.set_title("统计特征柱状图", fontsize=16, fontweight="bold", pad=10, color="#111827")
    ax2.set_ylabel(value_col, fontsize=12)
    ax2.set_xlabel("年龄组", fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"{g}岁" if g != "70+" else "70+岁" for g in age_order], rotation=18)
    ax2.axhline(0, color="#334155", lw=1.2, alpha=0.95)
    ax2.grid(which="major", axis="y", linestyle="--", color="#94a3b8", linewidth=0.8, alpha=0.9)
    ax2.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax2.grid(which="minor", axis="y", linestyle=":", color="#cbd5e1", linewidth=0.55, alpha=0.95)
    ax2.tick_params(axis="both", which="major", direction="in", length=5, width=1.1, labelsize=10.5)
    ax2.tick_params(axis="both", which="minor", direction="in", length=3, width=0.8)

    for spine in ["top", "right", "left", "bottom"]:
        ax2.spines[spine].set_visible(True)
        ax2.spines[spine].set_linewidth(1.2)
        ax2.spines[spine].set_color("#334155")

    lg2 = ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=2,
        frameon=True,
        facecolor="#ffffff",
        edgecolor="#475569",
        framealpha=1.0,
        fontsize=10.5,
    )
    lg2.get_frame().set_linewidth(1.0)

    # 显著性标注只画在 PeakValue 上，使用的是两两比较经过 Holm 校正后的结果。
    # 在峰值功率柱状图上标注显著性（基于 Holm 校正后的 p 值）
    sig_peak = signif_pairwise_df[
        (signif_pairwise_df["Metric"] == "PeakValue")
        & (signif_pairwise_df["significant_p<0.05"] == True)
    ].copy()
    sig_peak = sig_peak.sort_values("p_holm")

    age_to_idx = {str(g): i for i, g in enumerate(age_order)}
    peak_x = x_pos - bar_w / 2

    if not sig_peak.empty:
        y_max_bar = float(np.nanmax(peak_mean + peak_std))
        y_min_ax, y_max_ax = ax2.get_ylim()
        y_range = y_max_ax - y_min_ax
        base_h = y_max_bar + 0.05 * y_range
        step_h = 0.07 * y_range

        for layer, row in enumerate(sig_peak.itertuples(index=False)):
            g1 = str(row.Group1)
            g2 = str(row.Group2)
            if g1 not in age_to_idx or g2 not in age_to_idx:
                continue

            i1, i2 = age_to_idx[g1], age_to_idx[g2]
            x1, x2 = peak_x[min(i1, i2)], peak_x[max(i1, i2)]
            y = base_h + layer * step_h
            h = 0.02 * y_range
            stars = p_to_stars(float(row.p_holm))

            ax2.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="#111827", lw=1.1)
            ax2.text((x1 + x2) / 2, y + h + 0.008 * y_range, stars, ha="center", va="bottom", fontsize=11)

        top_needed = base_h + (len(sig_peak) - 1) * step_h + 0.12 * y_range
        if top_needed > y_max_ax:
            ax2.set_ylim(y_min_ax, top_needed)

    # 统一给右图额外顶部余量，避免显著性线和边框过近
    y_min2, y_max2 = ax2.get_ylim()
    ax2.set_ylim(y_min2, y_max2 + 0.04 * (y_max2 - y_min2))

    # 顶部留更多空间给大标题，底部留空间给图例，避免遮挡
    fig.tight_layout(rect=(0, 0.14, 1, 0.88), w_pad=2.2)
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni 多重比较校正。"""
    # Holm-Bonferroni 用于控制多重比较误差，避免成对检验过多时显著性被高估。
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted_sorted = [0.0] * m

    for rank, idx in enumerate(order):
        adjusted_sorted[rank] = (m - rank) * p_values[idx]

    # 保证校正后 p 值单调不减
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])

    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = min(adjusted_sorted[rank], 1.0)
    return adjusted


def run_significance_analysis(feature_by_subject: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 PeakValue / TroughValue 进行显著性分析并返回结果表。"""
    # 先做整体组间差异检验，再做两两比较，这样输出结果更适合汇总和解释。
    metrics = ["PeakValue", "TroughValue"]
    age_groups = sorted(feature_by_subject["AgeGroup"].dropna().unique(), key=str)

    global_rows = []
    pairwise_rows = []

    for metric in metrics:
        group_values = []
        for g in age_groups:
            values = feature_by_subject.loc[feature_by_subject["AgeGroup"] == g, metric].dropna().to_numpy()
            if len(values) > 0:
                group_values.append(values)

        if len(group_values) >= 2:
            f_stat, p_anova = stats.f_oneway(*group_values)
            h_stat, p_kruskal = stats.kruskal(*group_values)
        else:
            f_stat, p_anova, h_stat, p_kruskal = np.nan, np.nan, np.nan, np.nan

        global_rows.append(
            {
                "Metric": metric,
                "ANOVA_F": f_stat,
                "ANOVA_p": p_anova,
                "ANOVA_significant_p<0.05": bool(p_anova < 0.05) if pd.notna(p_anova) else False,
                "Kruskal_H": h_stat,
                "Kruskal_p": p_kruskal,
                "Kruskal_significant_p<0.05": bool(p_kruskal < 0.05) if pd.notna(p_kruskal) else False,
            }
        )

        raw_p = []
        temp_rows = []
        for g1, g2 in combinations(age_groups, 2):
            x = feature_by_subject.loc[feature_by_subject["AgeGroup"] == g1, metric].dropna().to_numpy()
            y = feature_by_subject.loc[feature_by_subject["AgeGroup"] == g2, metric].dropna().to_numpy()

            if len(x) < 2 or len(y) < 2:
                p_val = np.nan
                u_stat = np.nan
            else:
                u_stat, p_val = stats.mannwhitneyu(x, y, alternative="two-sided")

            temp_rows.append(
                {
                    "Metric": metric,
                    "Group1": str(g1),
                    "Group2": str(g2),
                    "n1": int(len(x)),
                    "n2": int(len(y)),
                    "MannWhitneyU": u_stat,
                    "p_raw": p_val,
                }
            )
            raw_p.append(1.0 if pd.isna(p_val) else float(p_val))

        adjusted = holm_adjust(raw_p) if raw_p else []
        for i, row in enumerate(temp_rows):
            p_adj = adjusted[i] if adjusted else np.nan
            row["p_holm"] = p_adj
            row["significant_p<0.05"] = bool(p_adj < 0.05) if pd.notna(p_adj) else False
            pairwise_rows.append(row)

    global_df = pd.DataFrame(global_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    return global_df, pairwise_df


def main() -> None:
    # 主流程：读取基础信息 -> 遍历右侧 sheet -> 合并 -> 统计 -> 导出结果和图像。
    demo_path = "data/01_Demo_PhysEx.xlsx"
    gait_path = "data/02_Overview_comf.xlsx"
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    demo = load_demo_info(demo_path)
    right_sheets = get_right_side_sheets(gait_path)
    if not right_sheets:
        raise ValueError("未在工作簿中识别到右侧肢体相关 sheet。")

    print("识别到以下右侧肢体相关 sheet:")
    for s in right_sheets:
        print(f"- {s}")

    success_sheets = []
    failed_sheets = []

    for sheet_name in right_sheets:
        print(f"\n开始处理: {sheet_name}")
        try:
            curve_long = load_curve_long(gait_path, sheet_name=sheet_name, value_col="Value")
            merged = curve_long.merge(demo, on="Subject", how="inner")
            merged = merged.dropna(subset=["AgeGroup"])

            # 此时 merged 是分析主表：每行表示某位受试者在某个步态周期点的一个数值。
            if merged.empty:
                print(f"  跳过：{sheet_name} 合并后无可用数据")
                failed_sheets.append((sheet_name, "合并后无可用数据"))
                continue

            group_n = merged[["Subject", "AgeGroup"]].drop_duplicates().groupby("AgeGroup", observed=True).size()
            # 先统计每组样本量，再把长表压缩成“每位受试者一行”的峰值/谷值特征表。
            feature_by_subject = (
                merged.groupby(["AgeGroup", "Subject"], observed=True)["Value"]
                .agg(PeakValue="max", TroughValue="min")
                .reset_index()
            )
            feature_summary = (
                feature_by_subject.groupby("AgeGroup", observed=True)[["PeakValue", "TroughValue"]]
                .agg(["mean", "std"])
            )
            signif_global_df, signif_pairwise_df = run_significance_analysis(feature_by_subject)

            output_png = str(output_dir / f"{sheet_name}_by_age_group.png")
            output_csv = str(output_dir / f"{sheet_name}_merged_long.csv")
            output_feature_csv = str(output_dir / f"{sheet_name}_feature_summary.csv")
            output_signif_global_csv = str(output_dir / f"{sheet_name}_significance_global.csv")
            output_signif_pairwise_csv = str(output_dir / f"{sheet_name}_significance_pairwise.csv")

            merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
            feature_summary.to_csv(output_feature_csv, encoding="utf-8-sig")
            signif_global_df.to_csv(output_signif_global_csv, index=False, encoding="utf-8-sig")
            signif_pairwise_df.to_csv(output_signif_pairwise_csv, index=False, encoding="utf-8-sig")

            plot_power_by_age_group(
                merged_long=merged,
                output_png=output_png,
                group_n=group_n,
                signif_pairwise_df=signif_pairwise_df,
                sheet_name=sheet_name,
                value_col="Value",
            )

            print("  各年龄段样本数:")
            print(group_n)
            print("  总体显著性检验:")
            print(signif_global_df.to_string(index=False))
            print(f"  已输出图像: {output_png}")
            success_sheets.append(sheet_name)

        except Exception as exc:  # noqa: BLE001
            failed_sheets.append((sheet_name, str(exc)))
            print(f"  处理失败: {sheet_name} -> {exc}")

    print("\n处理完成。")
    print(f"成功: {len(success_sheets)} 个")
    print(f"失败: {len(failed_sheets)} 个")
    if failed_sheets:
        print("失败明细:")
        for s, msg in failed_sheets:
            print(f"- {s}: {msg}")


if __name__ == "__main__":
    main()
