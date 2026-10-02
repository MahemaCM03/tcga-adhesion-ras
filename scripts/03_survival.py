"""
Author: Mahema CM
Project: TCGA Multi-Omic Adhesion & RAS Pathway Analysis
Script 03: Survival Modeling (Kaplan-Meier & Cox Proportional Hazards)

Goal:
  1. Load processed cohort data from Script 01 ('merged_alterations.csv').
  2. Filter out records missing time-to-event duration (OS_MONTHS) or vital status (OS_EVENT).
  3. Estimate Kaplan-Meier survival curves and perform non-parametric Log-Rank tests.
  4. Fit multivariate Cox Proportional Hazards models adjusting for clinical covariates (Age, Stage).
  5. Apply Benjamini-Hochberg FDR adjustments to control false discovery rates across tests.
  6. Generate Kaplan-Meier curves and a Cox model Forest Plot.
  7. Export statistical results to 'data/processed/survival_logrank_results.csv' 
     and 'data/processed/survival_cox_results.csv'.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend rendering for saving PNG files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from statsmodels.stats.multitest import multipletests

# Pathway names to distinguish composite flags from single genes
GROUP_NAMES = ["ADHESION", "RAS"]

# Statistical threshold cutoffs for subgroup survival modeling
MIN_GROUP = 10              # Minimum patients required in altered/wild-type groups
MIN_EVENTS_ALTERED = 5      # Minimum death events required in the altered group to ensure model stability
FDR_THRESHOLD = 0.05        # Benjamini-Hochberg significance cutoff


def alteration_columns(df: pd.DataFrame) -> list:
    """
    Extracts all genomic alteration flag headers, including single target genes 
    (e.g., CDH1, KRAS) and composite pathway flags (ADHESION, RAS).
    """
    genes = [col[4:] for col in df.columns if col.startswith("alt_") and col[4:] not in GROUP_NAMES]
    return genes + [group for group in GROUP_NAMES if f"alt_{group}" in df.columns]


def parse_stage(stage_text: str) -> float:
    """
    Converts text-based AJCC pathologic stage strings into ordinal numeric values for regression.
    Example: 'Stage IIA' -> 2.0, 'Stage IIIB' -> 3.0
    """
    if not isinstance(stage_text, str):
        return np.nan
    
    clean_stage = stage_text.upper().replace("STAGE", "").strip()
    if clean_stage.startswith("IV"):
        return 4.0
    if clean_stage.startswith("III"):
        return 3.0
    if clean_stage.startswith("II"):
        return 2.0
    if clean_stage.startswith("I"):
        return 1.0
    return np.nan


def load_survival_table(path: Path) -> pd.DataFrame:
    """
    Loads merged alteration data and keeps only records with complete, valid survival metadata.
    - OS_MONTHS: Follow-up time in months (must be > 0)
    - OS_EVENT: 1.0 = Deceased, 0.0 = Living (right-censored)
    """
    df = pd.read_csv(path)
    df = df.dropna(subset=["OS_MONTHS", "OS_EVENT"])
    df = df[df["OS_MONTHS"] > 0].copy()
    df["OS_EVENT"] = df["OS_EVENT"].astype(int)
    return df


def km_and_logrank(df: pd.DataFrame, alt_names: list, fig_dir: Path) -> pd.DataFrame:
    """
    Fits non-parametric Kaplan-Meier survival curves and computes Log-Rank test 
    p-values comparing altered vs. wild-type cohort groups.
    """
    rows = []
    
    for study_id, sub_df in df.groupby("study_id"):
        for alt_name in alt_names:
            
            # Separate cohort into altered (True) vs. wild-type (False)
            altered_flag = sub_df[f"alt_{alt_name}"].astype(bool)
            altered_group = sub_df[altered_flag]
            wildtype_group = sub_df[~altered_flag]

            row = {
                "study_id": study_id,
                "alteration": alt_name,
                "n_altered": len(altered_group),
                "n_unaltered": len(wildtype_group),
                "events_altered": int(altered_group["OS_EVENT"].sum()),
                "events_unaltered": int(wildtype_group["OS_EVENT"].sum()),
            }

            # Enforce sample size and death event safety checks before testing
            if len(altered_group) < MIN_GROUP or len(wildtype_group) < MIN_GROUP:
                row["note"] = f"Skipped: <{MIN_GROUP} patients in group"
            elif row["events_altered"] < MIN_EVENTS_ALTERED:
                row["note"] = f"Skipped: <{MIN_EVENTS_ALTERED} death events"
            else:
                # Log-Rank test evaluates if there is a significant difference between two survival curves
                lr_result = logrank_test(
                    altered_group["OS_MONTHS"],
                    wildtype_group["OS_MONTHS"],
                    event_observed_A=altered_group["OS_EVENT"],
                    event_observed_B=wildtype_group["OS_EVENT"],
                )
                row["p_value"] = lr_result.p_value
                row["note"] = "tested"

                # Generate Kaplan-Meier curve figure
                plot_km(
                    altered_group,
                    wildtype_group,
                    study_id,
                    alt_name,
                    fig_dir / f"km_{study_id}_{alt_name}.png",
                    lr_result.p_value,
                )

            rows.append(row)

    results_df = pd.DataFrame(rows)
    
    # Apply Benjamini-Hochberg FDR correction on tested p-values
    tested_mask = results_df["note"] == "tested"
    if tested_mask.any():
        results_df.loc[tested_mask, "q_value"] = multipletests(
            results_df.loc[tested_mask, "p_value"], method="fdr_bh"
        )[1]

    return results_df


def plot_km(
    altered_group: pd.DataFrame, 
    wildtype_group: pd.DataFrame, 
    study_id: str, 
    alt_name: str, 
    out_path: Path, 
    p_value: float
):
    """
    Plots and exports Kaplan-Meier overall survival curves with 95% confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Fit and plot wild-type curve
    kmf_wt = KaplanMeierFitter()
    kmf_wt.fit(
        wildtype_group["OS_MONTHS"], 
        wildtype_group["OS_EVENT"], 
        label=f"{alt_name} Wild-Type (n={len(wildtype_group)})"
    )
    kmf_wt.plot_survival_function(ax=ax, ci_show=True, color="#4c72b0")

    # Fit and plot altered curve
    kmf_alt = KaplanMeierFitter()
    kmf_alt.fit(
        altered_group["OS_MONTHS"], 
        altered_group["OS_EVENT"], 
        label=f"{alt_name} Altered (n={len(altered_group)})"
    )
    kmf_alt.plot_survival_function(ax=ax, ci_show=True, color="#c44e52")

    ax.set_xlabel("Months")
    ax.set_ylabel("Overall Survival Probability")
    ax.set_title(f"{study_id}\nLog-Rank p = {p_value:.3g}", fontsize=10)
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def cox_models(df: pd.DataFrame, alt_names: list) -> pd.DataFrame:
    """
    Fits multivariate Cox Proportional Hazards regression models adjusting for 
    patient age and tumor pathologic stage to calculate Hazard Ratios (HR).
    """
    df = df.copy()
    covariates = []

    # Check for clinical covariates
    if "AGE" in df.columns:
        df["AGE"] = pd.to_numeric(df["AGE"], errors="coerce")
        covariates.append("AGE")
        
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in df.columns:
        df["STAGE_NUM"] = df["AJCC_PATHOLOGIC_TUMOR_STAGE"].map(parse_stage)
        covariates.append("STAGE_NUM")

    rows = []
    
    for alt_name in alt_names:
        model_df = df[["OS_MONTHS", "OS_EVENT", "study_id"] + covariates].copy()
        model_df["altered"] = df[f"alt_{alt_name}"].astype(int)
        model_df = model_df.dropna()

        n_altered = int(model_df["altered"].sum())
        events_altered = int(model_df.loc[model_df["altered"] == 1, "OS_EVENT"].sum())

        row = {
            "alteration": alt_name,
            "n_used": len(model_df),
            "n_altered": n_altered,
            "events_altered": events_altered,
            "adjusted_for": ", ".join(covariates) + ", stratified by study",
        }

        # Safety check for cohort size and event count
        if n_altered < MIN_GROUP or events_altered < MIN_EVENTS_ALTERED:
            row["note"] = "Skipped: Group too small"
            rows.append(row)
            continue

        try:
            cph = CoxPHFitter()
            strata_cols = ["study_id"] if model_df["study_id"].nunique() > 1 else None
            
            if strata_cols is None:
                model_df = model_df.drop(columns="study_id")

            # Fit Cox proportional hazards model
            cph.fit(model_df, duration_col="OS_MONTHS", event_col="OS_EVENT", strata=strata_cols)
            
            # Extract summary statistics for the genomic alteration term
            summary_stats = cph.summary.loc["altered"]
            row.update({
                "HR": summary_stats["exp(coef)"],                  # Hazard Ratio
                "CI_low": summary_stats["exp(coef) lower 95%"],    # Lower 95% Confidence Interval limit
                "CI_high": summary_stats["exp(coef) upper 95%"],   # Upper 95% Confidence Interval limit
                "p_value": summary_stats["p"],                     # Wald Test p-value
                "note": "fitted",
            })
        except Exception as err:
            row["note"] = f"Model convergence failed: {err}"

        rows.append(row)

    results_df = pd.DataFrame(rows)
    
    # Apply Benjamini-Hochberg FDR correction on fitted models
    fitted_mask = results_df["note"] == "fitted"
    if fitted_mask.any():
        results_df.loc[fitted_mask, "q_value"] = multipletests(
            results_df.loc[fitted_mask, "p_value"], method="fdr_bh"
        )[1]

    return results_df


def forest_plot(cox_results: pd.DataFrame, out_path: Path):
    """
    Generates a Forest Plot summarizing Cox Model Hazard Ratios and 95% Confidence Intervals.
    - HR > 1.0: Associated with higher mortality risk
    - HR < 1.0: Associated with reduced mortality risk
    - CI crossing 1.0: Non-significant association
    """
    fitted_df = cox_results[cox_results["note"] == "fitted"].sort_values("HR")
    if fitted_df.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 0.5 * len(fitted_df) + 1.5))
    y_positions = np.arange(len(fitted_df))

    # Plot error bars representing Hazard Ratios and 95% Confidence Intervals
    ax.errorbar(
        fitted_df["HR"],
        y_positions,
        xerr=[fitted_df["HR"] - fitted_df["CI_low"], fitted_df["CI_high"] - fitted_df["HR"]],
        fmt="o",
        color="black",
        capsize=3,
    )

    # Reference line at HR = 1.0 (no effect)
    ax.axvline(1.0, color="grey", linestyle="--")
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(fitted_df["alteration"])
    ax.set_xscale("log")
    ax.set_xlabel("Hazard Ratio (95% CI), Log Scale\n(<1 protective, >1 elevated risk)")
    ax.set_title("Multivariate Cox PH Model (Adjusted for Age & Stage)", fontsize=10)
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

    survival_df = load_survival_table(merged_path)
    alterations_to_test = alteration_columns(survival_df)

    # Run Kaplan-Meier + Log-Rank tests
    km_results = km_and_logrank(survival_df, alterations_to_test, fig_dir)
    km_results.to_csv(out_dir / "survival_logrank_results.csv", index=False)

    # Run Multivariate Cox PH models
    cox_results = cox_models(survival_df, alterations_to_test)
    cox_results.to_csv(out_dir / "survival_cox_results.csv", index=False)
    forest_plot(cox_results, fig_dir / "cox_forest.png")

    print("Done. Survival analysis complete.")
    print("\nLog-Rank Results Summary:")
    print(km_results[["alteration", "n_altered", "events_altered", "p_value", "note"]].to_string(index=False))

    print("\nCox PH Model Results Summary:")
    print(cox_results[["alteration", "n_altered", "events_altered", "HR", "p_value", "note"]].to_string(index=False))


if __name__ == "__main__":
    main()