"""
Author: Mahema CM
Project: TCGA Multi-Omic Adhesion & RAS Pathway Analysis
Script 01: ETL, Multi-Omic Alignment, & Feature Engineering

Goal:
  1. Load raw cBioPortal TCGA Breast Cancer (BRCA) files.
  2. Filter out silent/unimportant mutations and retain non-silent functional mutations.
  3. Classify Copy Number Alterations (CNA) into Deep Deletions (-2) for Tumor Suppressors 
     or High Amplifications (+2) for Oncogenes.
  4. Perform a strict 3-way intersection across Clinical, Mutation, and CNA cohorts.
  5. Combine Mutations + CNA into binary pathway alteration flags.
  6. Apply a log2(RSEM + 1) transformation to normalize mRNA expression levels.
  7. Export the final merged dataset
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Define the biological role for each target gene
# 'tsg' = Tumor Suppressor Gene (cancer happens when we lose both copies, CNA <= -2)
# 'onc' = Oncogene (cancer happens when the gene is over-amplified, CNA >= 2)
GENE_ROLES = {
    "CDH1": "tsg",
    "CTNNA1": "tsg",
    "CTNNB1": "onc",
    "HRAS": "onc",
    "KRAS": "onc",
    "NRAS": "onc",
    "NF1": "tsg",
}

# Group individual genes into higher-level biological pathways.
# If ANY gene in a pathway is altered, the pathway flag becomes True (1).
GENE_GROUPS = {
    "ADHESION": ["CDH1", "CTNNA1", "CTNNB1"],
    "RAS": ["HRAS", "KRAS", "NRAS", "NF1"],
}

# Mutation types that actually change protein structure or function.
# Silent mutations (which do not change amino acids) are excluded.
NON_SILENT = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Translation_Start_Site",
    "Nonstop_Mutation",
}

# Extract the list of all individual gene symbols from our dictionary keys
GENES = list(GENE_ROLES)


def read_cbio_table(path: Path) -> pd.DataFrame:
    """
    Read a tab-separated cBioPortal file, skipping the leading '#' metadata lines.
    cBioPortal files start with comment lines that begin with '#'. 
    If pandas reads these lines directly, it throws an error. This function 
    counts how many '#' comment lines exist and skips them automatically.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    
    # Count how many lines at the top of the file start with '#'
    n_skip = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                n_skip += 1
            else:
                break

    # Load the table into a pandas DataFrame, skipping the comment lines        
    return pd.read_csv(path, sep="\t", skiprows=n_skip, low_memory=False)


def read_gene_by_sample_matrix(path: Path, genes) -> pd.DataFrame:
    """
    Loads Copy Number (CNA) or mRNA expression matrix files.
    
    In raw cBioPortal files:
      - Rows = Genes
      - Columns = Patient/Sample IDs
      
    This function filters for only our target genes and transposes (flips) 
    the matrix so that:
      - Rows = Patient/Sample IDs
      - Columns = Genes
    """
    df = read_cbio_table(path)

    # Keep only rows where the Hugo_Symbol is one of our target genes
    df = df[df["Hugo_Symbol"].isin(genes)].drop_duplicates("Hugo_Symbol")
    
    # Set the gene symbol as the row index, and drop unnecessary ID columns
    df = df.set_index("Hugo_Symbol").drop(columns=["Entrez_Gene_Id"], errors="ignore")
    
    # Convert string numbers to true numeric floats
    df = df.apply(pd.to_numeric, errors="coerce")
    
    # Transpose (.T) so rows are samples and columns are genes
    return df.T.reindex(columns=genes)


def read_sequenced_samples(study_dir: Path):
    """
    Reads the list of patient sample IDs that were ACTUALLY sequenced.

    The mutation file ('data_mutations.txt') ONLY lists patients who have a mutation. 
    It does NOT list unmutated patients. Without this file, you cannot distinguish 
    between a patient who has NO mutation vs. a patient who was NEVER sequenced!
    """
    path = study_dir / "case_lists" / "cases_sequenced.txt"
    if not path.exists():
        return None  # Fallback if case list file isn't present
   
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("case_list_ids:"):
                # Extract the sample IDs separated by tabs
                ids = line.split(":", 1)[1].strip().split("\t")
                return {i.strip() for i in ids if i.strip()}
    return None


def process_study(study_dir: Path):
    """
    Main Data Processing Engine:
    Integrates Clinical Metadata, Somatic Mutations, Copy-Number Variations, 
    and mRNA Expression into a single merged table.
    """  
    study_id = study_dir.name
    print(f"\n=== {study_id} ===")
    sizes = {"study_id": study_id}

    # Read and merge sample metadata with patient clinical records
    sample = read_cbio_table(study_dir / "data_clinical_sample.txt")
    patient = read_cbio_table(study_dir / "data_clinical_patient.txt")
    
    # Left join sample metadata with patient data on the PATIENT_ID column
    clin = sample.merge(patient, on="PATIENT_ID", how="left")
    sizes["n_clinical_samples"] = clin["SAMPLE_ID"].nunique()

    # Read Copy-Number Alteration (CNA) data
    cna = read_gene_by_sample_matrix(study_dir / "data_cna.txt", GENES)
    
    missing = [g for g in GENES if cna[g].isna().all()]
    if missing:
        print(f"  WARNING: no CNA data for {missing}")

    # Get the list of verified sequenced samples
    sequenced = read_sequenced_samples(study_dir)
    if sequenced is None:
        print("  NOTE: cases_sequenced.txt not found; using samples with CNA data only")
        sequenced = set(cna.index)

    # Perform a 3-way intersection (Keep ONLY samples that have Clinical Data AND Mutation Sequencing AND Copy-Number Data.)    
    profiled = set(clin["SAMPLE_ID"]) & sequenced & set(cna.index)
    sizes["n_profiled_mut_and_cna"] = len(profiled)

    # Filter the clinical dataframe to keep only profiled samples
    clin = clin[clin["SAMPLE_ID"].isin(profiled)].copy()

    # One sample per patient (keeps survival analysis honest: no patient counted twice)
    n_dupes = clin["PATIENT_ID"].duplicated().sum()
    if n_dupes:
        print(f"  {n_dupes} extra samples from the same patient dropped (kept first)")
        clin = clin.drop_duplicates("PATIENT_ID", keep="first")
    sizes["n_final"] = len(clin)
    clin = clin.set_index("SAMPLE_ID")
    samples = clin.index

    # --- mutations: non-silent calls per gene per sample --------------------------------
    mut = read_cbio_table(study_dir / "data_mutations.txt")
    
    # Filter for target genes and non-silent mutation classifications
    mut = mut[
        mut["Hugo_Symbol"].isin(GENES) & 
        mut["Variant_Classification"].isin(NON_SILENT)
    ]
    
    # Create a boolean matrix (True if mutated, False if wild-type)
    mut_flags = (
        mut.groupby(["Tumor_Sample_Barcode", "Hugo_Symbol"]).size().unstack(fill_value=0) > 0
    )
    mut_flags = mut_flags.reindex(index=samples, columns=GENES, fill_value=False)

    # Parse mRNA Expression Data and apply log2(x + 1) normalization.
    # Raw expression counts range from 0 to 100,000+. 
    # Log-transforming compresses extreme outliers, making data suitable for statistical testing.
    # We add +1 so that log2(0 + 1) = 0 (avoiding undefined log(0)).

    expr_path = study_dir / "data_mrna_seq_v2_rsem.txt"
    if expr_path.exists():
        expr = read_gene_by_sample_matrix(expr_path, GENES)
        expr = np.log2(expr.reindex(index=samples) + 1)
    else:
        print("  NOTE: no mRNA file found; expression columns will be empty")
        expr = pd.DataFrame(np.nan, index=samples, columns=GENES)

    # Assemble the output matrix
    out = pd.DataFrame(index=samples)
    out["study_id"] = study_id
    out["PATIENT_ID"] = clin["PATIENT_ID"]

    for gene, role in GENE_ROLES.items():
        gene_cna = cna[gene].reindex(samples)
        
        # TSGs require deep deletion (<= -2)
        # Oncogenes require high-level amplification (>= +2)
        cna_hit = (gene_cna <= -2) if role == "tsg" else (gene_cna >= 2)
        out[f"mut_{gene}"] = mut_flags[gene].values
        out[f"cna_{gene}"] = cna_hit.fillna(False).values
        out[f"alt_{gene}"] = out[f"mut_{gene}"] | out[f"cna_{gene}"]
        out[f"expr_{gene}"] = expr[gene].values

    # If ANY gene in the pathway group is altered, the pathway flag becomes True (1).
    for group, members in GENE_GROUPS.items():
        out[f"alt_{group}"] = out[[f"alt_{g}" for g in members]].any(axis=1)

    # Attach clinical columns (Age, Sex, Stage, Subtype, Survival Duration/Status)
    for col in ["CANCER_TYPE_ACRONYM", "CANCER_TYPE_DETAILED", "AGE", "SEX",
                "AJCC_PATHOLOGIC_TUMOR_STAGE", "SUBTYPE"]:
        if col in clin.columns:
            out[col] = clin[col]

    out["OS_MONTHS"] = pd.to_numeric(clin.get("OS_MONTHS"), errors="coerce")
    status = clin.get("OS_STATUS")
    if status is not None:
        # cBioPortal encodes this as "0:LIVING" / "1:DECEASED"
        event = status.astype("string").str.startswith("1").astype("float")
        out["OS_EVENT"] = event.where(status.notna())
    else:
        out["OS_EVENT"] = np.nan
        print("  WARNING: OS_STATUS column not found")

    return out, sizes


# --------------------------------------------------------------------------------------
# 4. SUMMARY TABLE
# --------------------------------------------------------------------------------------

def alteration_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """Computes summary statistics: how many patients had alterations in each 
    gene or pathway, both as raw counts and percentages."""
    rows = []
    for study, sub in df.groupby("study_id"):
        for gene in GENES + list(GENE_GROUPS):
            n_alt = int(sub[f"alt_{gene}"].sum())
            rows.append({
                "study_id": study,
                "gene_or_group": gene,
                "n_samples": len(sub),
                "n_altered": n_alt,
                "pct_altered": round(100 * n_alt / len(sub), 2),
                "n_mutated": int(sub[f"mut_{gene}"].sum()) if gene in GENES else np.nan,
                "n_cna": int(sub[f"cna_{gene}"].sum()) if gene in GENES else np.nan,
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    studies = sorted(p for p in data_dir.iterdir()
                     if p.is_dir() and (p / "data_clinical_sample.txt").exists())
    if not studies:
        raise SystemExit(f"No study folders found in {data_dir}. See the instructions at the top of this file.")

    tables, sizes = [], []
    for study_dir in studies:
        table, size = process_study(study_dir)
        tables.append(table)
        sizes.append(size)

    merged = pd.concat(tables)
    merged.index.name = "SAMPLE_ID"

    merged.to_csv(out_dir / "merged_alterations.csv")
    alteration_frequency(merged).to_csv(out_dir / "alteration_frequency.csv", index=False)
    pd.DataFrame(sizes).to_csv(out_dir / "cohort_sizes.csv", index=False)

    print("\nDone.")
    print(f"  merged table:  {out_dir / 'merged_alterations.csv'}  ({merged.shape[0]} samples)")
    print("\nAlteration frequency (%), top rows:")
    print(alteration_frequency(merged).sort_values("pct_altered", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()