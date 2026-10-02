"""
Author: Mahema CM
Project: TCGA Multi-Omic Adhesion & RAS Pathway Analysis
Script 02: Pairwise Co-Occurrence & Mutual Exclusivity Testing

Goal:
  1. Generate all unique, non-repeating gene pairs from target genes.
  2. Build a 2x2 contingency matrix for each gene pair across samples.
  3. Perform Fisher's Exact Test to compute Odds Ratios (OR) and raw p-values.
  4. Apply Haldane-Anscombe (+0.5) adjustment to prevent infinite log2 odds ratios.
  5. Apply Benjamini-Hochberg False Discovery Rate (FDR) adjustment to convert p-values to q-values.
  6. Generate and save a heat map plot displaying log2 odds ratios.
  7. Save statistical test results to 'data/processed/cooccurrence_results.csv'.
"""

import argparse
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Use non-interactive background rendering for saving PNG figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

# Pathway names to exclude when building individual gene-gene pairs
GROUP_NAMES = {"ADHESION", "RAS"}

# Thresholds for statistical testing
MIN_ALTERED = 5       # Minimum altered samples required per gene to run a test
FDR_THRESHOLD = 0.05  # Significance cutoff for FDR-adjusted q-values


def single_genes(df: pd.DataFrame) -> list:
    """
    Extracts individual gene names from columns starting with 'alt_',
    excluding composite pathway group flags ('ADHESION', 'RAS').
    """
    return [col[4:] for col in df.columns if col.startswith("alt_") and col[4:] not in GROUP_NAMES]


def run_pair_tests(df: pd.DataFrame, genes: list, min_altered: int) -> pd.DataFrame:
    """
    Constructs 2x2 contingency tables and calculates Fisher's Exact Tests 
    for every unique pair of target genes.
    """
    rows = []
    
    # Iterate through each study in the merged dataset
    for study_id, sub_df in df.groupby("study_id"):
        # Generate all unique 2-gene combinations (e.g., CDH1-KRAS, CDH1-NF1)
        for gene_a, gene_b in itertools.combinations(genes, 2):
            
            # Extract boolean series (True if altered, False if wild-type)
            altered_a = sub_df[f"alt_{gene_a}"].astype(bool)
            altered_b = sub_df[f"alt_{gene_b}"].astype(bool)

            # Skip testing if either gene has fewer than the minimum required altered samples
            if altered_a.sum() < min_altered or altered_b.sum() < min_altered:
                continue

            # Construct 2x2 contingency table cell counts:
            both_altered = int((altered_a & altered_b).sum())      # Both Gene A and Gene B altered
            a_only = int((altered_a & ~altered_b).sum())           # Only Gene A altered
            b_only = int((~altered_a & altered_b).sum())           # Only Gene B altered
            neither_altered = int((~altered_a & ~altered_b).sum()) # Neither gene altered

            # Fisher's Exact Test: ideal for small discrete sample counts where Chi-Square fails
            odds_ratio, p_val = fisher_exact([[both_altered, a_only], [b_only, neither_altered]])

            # Calculate log2 Odds Ratio with Haldane-Anscombe (+0.5) adjustment to avoid division by zero
            log2_or = np.log2(
                ((both_altered + 0.5) * (neither_altered + 0.5)) / 
                ((a_only + 0.5) * (b_only + 0.5))
            )

            rows.append({
                "study_id": study_id, 
                "gene_a": gene_a, 
                "gene_b": gene_b,
                "n_samples": len(sub_df), 
                "n_a": int(altered_a.sum()), 
                "n_b": int(altered_b.sum()),
                "both": both_altered, 
                "a_only": a_only, 
                "b_only": b_only, 
                "neither": neither_altered,
                "odds_ratio": odds_ratio, 
                "log2_or": log2_or, 
                "p_value": p_val,
            })

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        return results_df

    # Multiple testing correction: Benjamini-Hochberg FDR procedure
    # Converts raw p-values into adjusted q-values to prevent false positives when running multiple tests
    results_df["q_value"] = multipletests(results_df["p_value"], method="fdr_bh")[1]
    
    # Label direction based on log2 Odds Ratio:
    # log2_or > 0 -> Co-occurrence (tumors tend to alter both genes together)
    # log2_or < 0 -> Mutual Exclusivity (tumors tend to alter only one of the two genes)
    results_df["direction"] = np.where(results_df["log2_or"] > 0, "co-occurrence", "mutual exclusivity")
    results_df["significant"] = results_df["q_value"] < FDR_THRESHOLD
    
    return results_df.sort_values("q_value").reset_index(drop=True)


def plot_heatmap(results_subset: pd.DataFrame, genes: list, study_id: str, out_path: Path):
    """
    Generates a symmetric heatmap plotting log2 odds ratios between tested gene pairs.
    Statistically significant pairs (q < 0.05) receive an asterisk (*).
    """
    # Initialize empty square matrices for values and annotations
    matrix_df = pd.DataFrame(np.nan, index=genes, columns=genes)
    annotation_df = pd.DataFrame("", index=genes, columns=genes)

    # Fill matrix with log2 odds ratios and significance markers
    for row in results_subset.itertuples():
        matrix_df.loc[row.gene_a, row.gene_b] = matrix_df.loc[row.gene_b, row.gene_a] = row.log2_or
        mark = "*" if row.q_value < FDR_THRESHOLD else ""
        annotation_df.loc[row.gene_a, row.gene_b] = annotation_df.loc[row.gene_b, row.gene_a] = mark

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    
    # Plot heatmap using Seaborn
    sns.heatmap(
        matrix_df, 
        annot=annotation_df, 
        fmt="", 
        cmap="coolwarm", 
        center=0, 
        vmin=-4, 
        vmax=4,
        linewidths=0.5, 
        linecolor="white", 
        cbar_kws={"label": "log2 odds ratio"},
        mask=matrix_df.isna(), 
        ax=ax, 
        annot_kws={"size": 16}
    )

    ax.set_title(
        f"{study_id}\nred = co-occurrence, blue = mutual exclusivity  (* FDR < {FDR_THRESHOLD})",
        fontsize=10
    )
    ax.set_facecolor("#eeeeee")  # Grey background indicates untested gene pairs (due to small sample sizes)
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
    parser.add_argument("--min-altered", type=int, default=MIN_ALTERED)
    args = parser.parse_args()

    merged_path = Path(args.merged)
    if not merged_path.exists():
        raise FileNotFoundError(f"Missing input file: {merged_path}. Please run Script 01 first.")

    merged_df = pd.read_csv(merged_path)
    target_genes = single_genes(merged_df)

    out_dir, fig_dir = Path(args.out_dir), Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Run pairwise tests
    results_df = run_pair_tests(merged_df, target_genes, args.min_altered)
    if results_df.empty:
        raise SystemExit("No gene pair had enough altered samples to perform testing.")

    # Save CSV results
    results_df.to_csv(out_dir / "cooccurrence_results.csv", index=False)

    # Generate heatmaps per study
    for study_id, sub_results in results_df.groupby("study_id"):
        plot_heatmap(sub_results, target_genes, study_id, fig_dir / f"cooccurrence_{study_id}.png")

    print(f"Done. Tested {len(results_df)} gene pairs.")
    print(f"Significant pairs at FDR < {FDR_THRESHOLD}: {int(results_df['significant'].sum())}")
    
    display_cols = ["study_id", "gene_a", "gene_b", "both", "a_only", "b_only", "odds_ratio", "q_value", "direction"]
    print("\nTop Pairwise Results:")
    print(results_df[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()