"""Tasks 7 and 8: the cleaned table and the weighted descriptive helpers.

Builds on `prepare_table` from clean.py. Adds height and weight, parsed
only far enough to audit BMI (they stay descriptive-only, see
reports/contract.md), and the survey weight for weighted rates and means.

Run from the project root to write data/processed/clean.csv:
  python src/diabetes_risk/describe.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from diabetes_risk.clean import prepare_table

BASE_DIR = Path(__file__).resolve().parent.parent.parent
THIN_TABLE_PATH = BASE_DIR / "data" / "processed" / "brfss2024_thin.parquet"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "clean.csv"

# Survey weight named in reports/data_dictionary.md.
WEIGHT_COLUMN = "_LLCPWT"

# Fixed class order and colors for every table and figure.
CLASS_ORDER = ["no_diabetes", "prediabetes", "diabetes"]
CLASS_COLORS = {"no_diabetes": "#2a78d6", "prediabetes": "#eb6834", "diabetes": "#1baf7a"}

AGE_BANDS = [18, 30, 40, 50, 60, 70, 81]
AGE_LABELS = ["18-29", "30-39", "40-49", "50-59", "60-69", "70-80"]

# Standard adult BMI groups (kg/m2).
BMI_BANDS = [0, 18.5, 25, 30, 35, 40, np.inf]
BMI_LABELS = ["under 18.5", "18.5-24.9", "25-29.9", "30-34.9", "35-39.9", "40+"]

CM_PER_INCH = 2.54
KG_PER_POUND = 0.45359237


def height_cm(series):
    """HEIGHT3 in centimetres.

    200-711 is feet and inches (507 -> 5 ft 7 in), 9000-9998 is
    centimetres plus 9000. 7777, 9999, and blank become missing.
    See reports/data_dictionary.md.
    """
    codes = pd.to_numeric(series, errors="coerce").round()
    feet = codes // 100
    inches = codes % 100
    imperial = ((feet * 12 + inches) * CM_PER_INCH).where(
        codes.between(200, 711) & (inches < 12)
    )
    metric = (codes - 9000).where(codes.between(9000, 9998))
    return imperial.combine_first(metric)


def weight_kg(series):
    """WEIGHT2 in kilograms.

    50-776 is pounds, 9000-9998 is kilograms plus 9000. 7777, 9999,
    and blank become missing. See reports/data_dictionary.md.
    """
    codes = pd.to_numeric(series, errors="coerce").round()
    pounds = (codes * KG_PER_POUND).where(codes.between(50, 776))
    metric = (codes - 9000).where(codes.between(9000, 9998))
    return pounds.combine_first(metric)


def build_clean_table(df):
    """`prepare_table` plus height, weight, and the survey weight.

    Height and weight are for auditing BMI and the descriptive charts
    only. They are not model inputs.
    """
    out = prepare_table(df)
    out["height_cm"] = height_cm(df["HEIGHT3"]).round(1)
    out["weight_kg"] = weight_kg(df["WEIGHT2"]).round(1)
    out["survey_weight"] = pd.to_numeric(df[WEIGHT_COLUMN], errors="coerce")
    return out


def weighted_mean(values, weights):
    """Survey-weighted mean, skipping rows where the value is missing."""
    values = pd.to_numeric(values, errors="coerce").astype(float)
    keep = values.notna() & weights.notna()
    if not keep.any():
        return np.nan
    return float(np.average(values[keep], weights=weights[keep]))


def weighted_share(flags, weights):
    """Survey-weighted share of True among answered rows (0 to 1)."""
    keep = flags.notna() & weights.notna()
    if not keep.any():
        return np.nan
    answered = flags[keep].astype(bool)
    return float(weights[keep][answered].sum() / weights[keep].sum())


def weighted_distribution(groups, weights):
    """Survey-weighted share of each group (sums to 1 over answered rows)."""
    keep = groups.notna() & weights.notna()
    totals = weights[keep].groupby(groups[keep], observed=False).sum()
    return totals / totals.sum()


def main():
    if not THIN_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Thin table not found at {THIN_TABLE_PATH}. Run src/diabetes_risk/load.py first."
        )
    df = pd.read_parquet(THIN_TABLE_PATH)
    clean = build_clean_table(df)

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)

    print(f"Rows: {len(clean):,}  Columns: {len(clean.columns)}")
    print(f"Saved {CLEAN_PATH}")


if __name__ == "__main__":
    main()
