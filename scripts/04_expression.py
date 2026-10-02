"""
Author: Mahema CM
Project: TCGA Multi-Omic Adhesion & RAS Pathway Analysis
Script 04: Transcriptomic Differential Expression (Mann-Whitney U Test & Cliff's Delta)

Goal:
  1. Load processed cohort data from Script 01 ('merged_alterations.csv').
  2. Extract normalized mRNA expression levels (log2(RSEM + 1)).
  3. Compare mRNA expression between genomic altered vs. wild-type tumors.
  4. Perform non-parametric Mann-Whitney U tests (ideal for skewed distributions & small groups).
  5. Calculate Cliff's Delta effect sizes to measure non-parametric magnitude of change.
  6. Apply Benjamini-Hochberg FDR adjustments across evaluated comparisons.
  7. Generate boxplots stratified by alteration mechanism and differential expression heatmaps.
  8. Export statistical results to 'data/processed/expression_results.csv'.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend rendering for saving PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Pathway names to separate composite flags from single target genes
GROUP_NAMES = ["ADHESION", "RAS"]

# Statistical threshold cutoffs
MIN_GROUP = 10        # Minimum required patients in altered/wild-type groups
FDR_THRESHOLD = 0.05  # Benjamini-Hochberg significance cutoff

# Order of mechanisms displayed on boxplots
CATEGORY_ORDER = ["unaltered", "mutation only", "CNA only", "mutation + CNA"]


def gene_names(df: pd.DataFrame) -> list:
    """
    Extracts individual gene names that have corresponding mRNA expression columns ('expr_').
    """
    return [col[5:] for col in df.columns if col.startswith("expr_")]


def alteration_names(df: pd.DataFrame) -> list:
    """
    Extracts all genomic alteration flag headers ('alt_'), including single genes and pathways.
    """
    genes = [col[4:] for col in df.columns if col.startswith("alt_") and col[4:] not in GROUP_NAMES]
    return genes + [group for group in GROUP_NAMES if f"alt_{group}" in df.columns]


def run_tests(df: pd.DataFrame, alt_names: list, expr_genes: list) -> pd.DataFrame:
    """
    Evaluates non-parametric differential expression using Mann-Whitney U tests and Cliff's Delta.
    Compares log2 mRNA expression between altered vs. wild-type tumors for every gene pair.
    """
    rows = []

    for study_id, sub_df in df.groupby("study_id"):
        for alt_name in alt_names:
            
            # Identify tumors with vs. without the genomic alteration
            altered_flag = sub_df[f"alt_{alt_name}"].astype(bool)

            for expr_gene in expr_genes:
                expr_column = f"expr_{expr_gene}"
                
                # Extract clean mRNA expression vectors
                altered_expr = sub_df.loc[altered_flag, expr_column].dropna()
                wildtype_expr = sub_df.loc[~altered_flag, expr_column].dropna()

                # Skip comparison if either group has fewer than the minimum required samples
                if len(altered_expr) < MIN_GROUP or len(wildtype_expr) < MIN_GROUP:
                    continue

                # Mann-Whitney U Test: Non-parametric alternative to independent two-sample t-test
                # Does not assume normal distribution of expression counts
                u_stat, p_val = mannwhitneyu(altered_expr, wildtype_expr, alternative="two-sided")

                # Compute Cliff's Delta effect size metric: ranges from -1.0 to +1.0
                # Delta > 0 indicates expression upregulation in altered tumors
                # Delta < 0 indicates expression downregulation in altered tumors
                cliffs_delta = (2 * u_stat / (len(altered_expr) * len(wildtype_expr))) - 1

                rows.append({
                    "study_id": study_id,
                    "alteration": alt_name,
                    "expression_gene": expr_gene,
                    "is_own_gene": alt_name == expr_gene,
                    "n_altered": len(altered_expr),
                    "n_unaltered": len(wildtype_expr),
                    "median_altered": altered_expr.median(),
                    "median_unaltered": wildtype_expr.median(),
                    "median_diff": altered_expr.median() - wildtype_expr.median(),
                    "cliffs_delta": cliffs_delta,
                    "p_value": p_val,
                })

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    # Benjamini-Hochberg FDR multiple testing adjustment
    results_df["q_value"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
    results_df["significant"] = results_df["q_value"] < FDR_THRESHOLD

    return results_df.sort_values("q_value").reset_index(drop=True)


def alteration_category(sub_df: pd.DataFrame, gene_symbol: str) -> pd.Series:
    """
    Categorizes each patient tumor sample into specific discrete alteration mechanisms:
      - 'unaltered': No non-silent mutation or focal CNA hit
      - 'mutation only': Point mutation present without deep deletion / high amplification
      - 'CNA only': Deep deletion or high amplification present without point mutation
      - 'mutation + CNA': Co-occurring point mutation AND copy number hit
    """
    mut_flag = sub_df[f"mut_{gene_symbol}"].astype(bool)
    cna_flag = sub_df[f"cna_{gene_symbol}"].astype(bool)

    conditions = [
        mut_flag & cna_flag,
        mut_flag & ~cna_flag,
        ~mut_flag & cna_flag,
    ]
    labels = ["mutation + CNA", "mutation only", "CNA only"]

    return pd.Series(
        np.select(conditions, labels, default="unaltered"),
        index=sub_df.index,
    )


def plot_own_gene(sub_df: pd.DataFrame, expr_genes: list, study_id: str, out_path: Path):
    """
    Generates boxplots showing target gene mRNA expression across specific alteration mechanisms.
    Validates biological controls (e.g., CDH1 deletion drops expression, KRAS amplification boosts it).
    """
    ncols = 4
    nrows = int(np.ceil(len(expr_genes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)

    for ax, gene_symbol in zip(axes.ravel(), expr_genes):
        plot_df = pd.DataFrame({
            "category": alteration_category(sub_df, gene_symbol),
            "mRNA": sub_df[f"expr_{gene_symbol}"],
        }).dropna()

        category_counts = plot_df["category"].value_counts()
        present_order = [cat for cat in CATEGORY_ORDER if category_counts.get(cat, 0) > 0]

        # Draw main distribution boxplot
        sns.boxplot(
            data=plot_df,
            x="category",
            y="mRNA",
            order=present_order,
            showfliers=False,
            color="#dddddd",
            ax=ax,
        )
        
        # Overlay individual sample data points
        sns.stripplot(
            data=plot_df,
            x="category",
            y="mRNA",
            order=present_order,
            size=2.5,
            alpha=0.5,
            color="#333333",
            ax=ax,
        )

        ax.set_xticks(range(len(present_order)))
        ax.set_xticklabels(
            [f"{cat}\n(n={category_counts[cat]})" for cat in present_order], 
            fontsize=7
        )
        ax.set_xlabel("")
        ax.set_ylabel("log2(RSEM + 1)")
        ax.set_title(gene_symbol, fontsize=10)

    # Turn off unused subplot axes
    for ax in axes.ravel()[len(expr_genes):]:
        ax.axis("off")

    fig.suptitle(f"{study_id}: mRNA Expression by Alteration Mechanism", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_heatmap(
    results_subset: pd.DataFrame, 
    alt_names: list, 
    expr_genes: list, 
    study_id: str, 
    out_path: Path
):
    """
    Generates a heatmap displaying median expression differences across evaluated genomic alterations.
    Statistically significant differences (q < 0.05) are marked with an asterisk (*).
    """
    matrix_df = pd.DataFrame(np.nan, index=alt_names, columns=expr_genes)
    annotation_df = pd.DataFrame("", index=alt_names, columns=expr_genes)

    for row in results_subset.itertuples():
        matrix_df.loc[row.alteration, row.expression_gene] = row.median_diff
        mark = "*" if row.q_value < FDR_THRESHOLD else ""
        annotation_df.loc[row.alteration, row.expression_gene] = mark

    fig, ax = plt.subplots(figsize=(1.0 * len(expr_genes) + 3, 0.5 * len(alt_names) + 2.5))
    
    sns.heatmap(
        matrix_df,
        annot=annotation_df,
        fmt="",
        cmap="coolwarm",
        center=0,
        mask=matrix_df.isna(),
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 16},
        cbar_kws={"label": "Median Difference in log2 mRNA\n(Altered - Wild-Type)"},
        ax=ax,
    )

    ax.set_facecolor("#eeeeee")
    ax.set_xlabel("mRNA Measured")
    ax.set_ylabel("Genomic Alteration")
    ax.set_title(f"{study_id} (* FDR < {FDR_THRESHOLD})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    """
    Command Line Interface (CLI) Execution Entry Point.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged", default="data/processed/merged_alterations.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--fig-dir", default="reports/figures")
    args = parser.parse_args()

    merged_path = Path(args.merged)
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing input file: {merged_path}. Please run Script 01 first.")

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    merged_df = pd.read_csv(merged_path)
    expression_genes = gene_names(merged_df)
    target_alterations = alteration_names(merged_df)

    # Run Mann-Whitney U differential expression pipeline
    results_df = run_tests(merged_df, target_alterations, expression_genes)
    if results_df.empty:
        raise SystemExit("No alteration-expression comparisons met sample size requirements.")

    # Export CSV results
    results_df.to_csv(out_dir / "expression_results.csv", index=False)

    # Generate figures per study
    for study_id, sub_df in merged_df.groupby("study_id"):
        plot_own_gene(sub_df, expression_genes, study_id, fig_dir / f"expr_owngene_{study_id}.png")
        plot_heatmap(
            results_df[results_df["study_id"] == study_id],
            target_alterations,
            expression_genes,
            study_id,
            fig_dir / f"expr_heatmap_{study_id}.png",
        )

    print(f"Done. Tested {len(results_df)} expression comparisons.")
    print(f"Significant expression changes at FDR < {FDR_THRESHOLD}: {int(results_df['significant'].sum())}")

    display_cols = [
        "study_id", "alteration", "expression_gene", "n_altered", 
        "median_diff", "cliffs_delta", "p_value", "q_value"
    ]
    print("\nTop Differential Expression Results:")
    print(results_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
    