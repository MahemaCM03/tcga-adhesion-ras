"""
Author: Mahema CM
Project: TCGA Multi-Omic Adhesion & RAS Pathway Analysis
Script 05: Histology Stratification, Within-Lobular Sensitivity, & Cross-Gene DEGs
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu


def main():
    merged_path = Path("data/processed/merged_alterations.csv")
    expr_path = Path("data/processed/expression_results.csv")
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not merged_path.exists():
        raise FileNotFoundError("Missing data/processed/merged_alterations.csv. Please run Script 01 first.")

    df = pd.read_csv(merged_path)

    print("==================================================")
    print(" 1. HISTOLOGY CHECK (CDH1 in Invasive Lobular)")
    print("==================================================")

    hist_col = "CANCER_TYPE_DETAILED" if "CANCER_TYPE_DETAILED" in df.columns else None

    if hist_col:
        # 2x2 Fisher's Exact Test: Lobular vs Non-Lobular
        df["is_lobular"] = df[hist_col].astype(str).str.contains("Lobular", case=False, na=False)
        tab = pd.crosstab(df["is_lobular"], df["alt_CDH1"])

        if tab.shape == (2, 2):
            both = tab.loc[True, True] if True in tab.index and True in tab.columns else 0
            lob_wt = tab.loc[True, False] if True in tab.index and False in tab.columns else 0
            nonlob_alt = tab.loc[False, True] if False in tab.index and True in tab.columns else 0
            nonlob_wt = tab.loc[False, False] if False in tab.index and False in tab.columns else 0

            odds_ratio, p_value = fisher_exact([[both, lob_wt], [nonlob_alt, nonlob_wt]])

            print(f"Lobular CDH1 Altered: {both} / {both + lob_wt} ({round(100 * both / (both + lob_wt), 2)}%)")
            print(f"Non-Lobular CDH1 Altered: {nonlob_alt} / {nonlob_alt + nonlob_wt} ({round(100 * nonlob_alt / (nonlob_alt + nonlob_wt), 2)}%)")
            print(f"Odds Ratio: {odds_ratio:.3f}")
            print(f"p-value: {p_value:.3e}\n")

        # --- WITHIN-LOBULAR SENSITIVITY CHECK ---
        print("--- WITHIN-LOBULAR SENSITIVITY CHECK ---")
        lob = df[df[hist_col] == "Breast Invasive Lobular Carcinoma"].copy()
        print(f"Cohort: Invasive Lobular Carcinoma only (N = {len(lob)})")
        print(f"CDH1 Altered: {lob['alt_CDH1'].sum()} | CDH1 Wild-Type: {(~lob['alt_CDH1']).sum()}\n")

        for gene in ["CDH1", "NRAS"]:
            altered_expr = lob.loc[lob["alt_CDH1"].astype(bool), f"expr_{gene}"].dropna()
            wildtype_expr = lob.loc[~lob["alt_CDH1"].astype(bool), f"expr_{gene}"].dropna()

            med_diff = altered_expr.median() - wildtype_expr.median()
            u_stat, p_val = mannwhitneyu(altered_expr, wildtype_expr, alternative="two-sided")

            print(f"Gene: {gene}")
            print(f"  Median Diff (Altered - WT): {med_diff:.4f}")
            print(f"  Mann-Whitney U p-value:    {p_val:.3e}")

        # --- GENERATE HISTOLOGY BAR CHART ---
        hist_summary = (
            df.groupby(hist_col)["alt_CDH1"]
            .agg(n_total="count", n_alt="sum")
            .reset_index()
        )
        hist_summary["pct_alt"] = (hist_summary["n_alt"] / hist_summary["n_total"]) * 100
        hist_summary = hist_summary[hist_summary["n_total"] >= 10].sort_values("pct_alt", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.barh(hist_summary[hist_col], hist_summary["pct_alt"], color="#c44e52")

        for bar, n_alt, n_tot in zip(bars, hist_summary["n_alt"], hist_summary["n_total"]):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.1f}% ({n_alt}/{n_tot})", 
                    va="center", fontsize=9, fontweight="bold")

        ax.set_xlabel("CDH1 Alteration Frequency (%)", fontsize=10)
        ax.set_title("CDH1 Alteration Rate by Histological Subtype (TCGA-BRCA, N = 1,052)", fontsize=11)
        ax.set_xlim(0, 70)
        ax.invert_yaxis()
        fig.tight_layout()

        bar_chart_path = fig_dir / "cdh1_histology_alteration_rate.png"
        fig.savefig(bar_chart_path, dpi=200)
        plt.close(fig)
        print(f"\nSaved histology bar chart to: {bar_chart_path}")

    print("\n==================================================")
    print(" 2. MOLECULAR SUBTYPE CHECK (KRAS Amplification)")
    print("==================================================")

    if "SUBTYPE" in df.columns:
        kras_pct = pd.crosstab(df["SUBTYPE"], df["alt_KRAS"], normalize="index").round(3) * 100
        print("--- KRAS Alterations by Subtype (%) ---")
        print(kras_pct)

    print("\n==================================================")
    print(" 3. CROSS-GENE DIFFERENTIAL EXPRESSION LOOKUP")
    print("==================================================")

    if expr_path.exists():
        expr_res = pd.read_csv(expr_path)
        cross_degs = expr_res[(expr_res["significant"] == True) & (expr_res["is_own_gene"] == False)]

        print(f"Total significant cross-gene comparisons (q < 0.05): {len(cross_degs)}\n")
        if not cross_degs.empty:
            cols_to_print = ["alteration", "expression_gene", "median_diff", "cliffs_delta", "p_value", "q_value"]
            print(cross_degs[cols_to_print].to_string(index=False))

if __name__ == "__main__":
    main()