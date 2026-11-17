"""Week 4: numeric modeling table built only from the frozen split.

The split is the one Task 6 already saved. Nothing here draws a new
random split. Medians, modes, and the decision to drop a column are
taken from the training rows only, then applied to validation and test.

Run from the repository root:

    PYTHONPATH=src python -m diabetes_risk.features
"""
from pathlib import Path

import numpy as np
import pandas as pd

from diabetes_risk.clean import THIN_TABLE_PATH, prepare_table
from diabetes_risk.config import SPLIT_SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC
from diabetes_risk.describe import AGE_BANDS, AGE_LABELS, BMI_BANDS, BMI_LABELS
from diabetes_risk.evaluate import LABELS
from diabetes_risk.split import SPLIT_PATH

ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = ROOT / "data" / "processed" / "features.csv"
KEYS_PATH = ROOT / "data" / "processed" / "feature_keys.csv"

# Ordered answers. The number goes up with current use, so one column
# is enough and a one-hot expansion is not needed.
SMOKER_ORD = {
    "never": 0,
    "former": 1,
    "current_some_days": 2,
    "current_every_day": 3,
}
SMOKELESS_ORD = {"not_at_all": 0, "some_days": 1, "every_day": 2}
ECIG_ORD = {"never": 0, "former": 1, "some_days": 2, "every_day": 3}

# A long tail of rare levels is collapsed only when a column actually
# has many levels. These survey items have two to four real answers.
HIGH_CARDINALITY = 10
RARE_SHARE = 0.01
NEAR_DUPLICATE = 0.98

SPLIT_CODE = {"train": 0, "validation": 1, "test": 2}
TARGET_CODE = {label: index for index, label in enumerate(LABELS)}

# Continuous columns a distance-based model should scale. Trees skip this.
CONTINUOUS = ["age", "bmi", "sugar_drinks", "bmi_to_age"]

# One line each: name, how it is computed, why it is in the model.
FEATURE_NOTES = {
    "age": (
        "Years from _AGE80. Missing ages use the training median.",
        "People with diabetes are older (weighted mean 60.5 versus 47.1).",
    ),
    "bmi": (
        "_BMI5 divided by 100. Missing BMI uses the training median.",
        "BMI rises from 27.9 with no diabetes to 31.8 with diabetes.",
    ),
    "sugar_drinks": (
        "SSBSUGR2 converted to times per day by the shared cleaner.",
        "Sugary soda is one of the contract inputs.",
    ),
    "any_exercise": (
        "EXERANY2 as 1 for yes and 0 for no. Missing answers use the training mode.",
        "Exercise is lower with diabetes (68%) than without it (81%).",
    ),
    "heavy_drinker": (
        "_RFDRHV9 as 1 for heavy drinking and 0 otherwise. Missing answers use the training mode.",
        "Heavy drinking is a contract input and a metabolic marker.",
    ),
    "smoker_ord": (
        "never 0, former 1, some days 2, every day 3.",
        "Cigarette use has an order, so one column replaces four dummies.",
    ),
    "smokeless_ord": (
        "not at all 0, some days 1, every day 2.",
        "Smokeless tobacco has the same kind of order.",
    ),
    "ecigarette_ord": (
        "never 0, former 1, some days 2, every day 3.",
        "Vaping has the same kind of order.",
    ),
    "sex_male": (
        "1 for male and 0 for female, after filling missing sex with the training mode.",
        "Sex has two unordered levels, so one dummy is enough.",
    ),
    "age_band": (
        "0 to 5 for 18-29, 30-39, 40-49, 50-59, 60-69, and 70-80. Same cuts as the descriptive charts.",
        "A straight age slope misses the midlife jump in diabetes.",
    ),
    "bmi_band": (
        "0 to 5 for the WHO groups under 18.5, 18.5-24.9, 25-29.9, 30-34.9, 35-39.9, and 40+.",
        "Diabetes concentrates at BMI 30 and above, which is how the charts already group body size.",
    ),
    "bmi_to_age": (
        "BMI divided by age, after the training-median fill.",
        "The same BMI at 30 and at 75 is not the same profile. The ratio is high when a high BMI shows up young.",
    ),
}


def _band_code(series, bins):
    """Ordinal band. The cut points are fixed, not learned from any split."""
    return pd.cut(series, bins=bins, labels=False, right=False).astype(float)


def _mode(series):
    modes = series.dropna().mode()
    if modes.empty:
        raise ValueError("cannot impute a column that is entirely missing on the training rows")
    return modes.iloc[0]


def collapse_rare_levels(series, train_mask, min_levels=HIGH_CARDINALITY, rare_share=RARE_SHARE):
    """Group rare levels into ``other`` only for high-cardinality columns.

    The decision uses training rows only. Ordinal survey items never reach
    this: they have fewer than ``min_levels`` answers.
    """
    levels = series.dropna().nunique()
    if levels < min_levels:
        return series, []
    shares = series.loc[train_mask].dropna().value_counts(normalize=True)
    rare = sorted(shares[shares < rare_share].index.tolist())
    if not rare:
        return series, []
    return series.where(~series.isin(rare), other="other"), rare


def drop_constants(frame, train_mask):
    """Drop a column that has only one training value. It cannot separate classes."""
    dropped = []
    kept = frame.copy()
    for column in frame.columns:
        if kept.loc[train_mask, column].nunique(dropna=False) <= 1:
            dropped.append(column)
            kept = kept.drop(columns=column)
    return kept, dropped


def drop_near_duplicates(frame, train_mask, threshold=NEAR_DUPLICATE):
    """Drop the later column when two training columns correlate at or above the threshold.

    Original measures sit to the left of the derived bands and the ratio,
    so the derived copy is the one that is removed.
    """
    kept = list(frame.columns)
    corr = frame.loc[train_mask, kept].corr().abs()
    to_drop = []
    details = []
    for index, left in enumerate(kept):
        if left in to_drop:
            continue
        for right in kept[index + 1:]:
            if right in to_drop:
                continue
            value = corr.loc[left, right]
            if pd.notna(value) and float(value) >= threshold:
                to_drop.append(right)
                details.append({
                    "column": right,
                    "duplicate_of": left,
                    "correlation": float(value),
                })
    return frame.drop(columns=to_drop), details


def _join_split(prepared, split):
    if "row_id" not in prepared.columns:
        raise ValueError("prepared table is missing row_id")
    labeled = prepared.loc[prepared["target"].notna()].copy()
    unknown = set(split["split"].unique()) - set(SPLIT_CODE)
    if unknown:
        raise ValueError(f"unexpected split names: {sorted(unknown)}")
    merged = labeled.merge(split, on="row_id", how="inner", validate="one_to_one")
    missing = set(SPLIT_CODE) - set(merged["split"].unique())
    if missing:
        raise ValueError(f"labeled rows are missing splits: {sorted(missing)}")
    return merged.sort_values("row_id").reset_index(drop=True)


def build_encoded(prepared, split):
    """Return the numeric table, the row keys, and a short build report.

    ``prepared`` is the output of ``prepare_table``. ``split`` has
    ``row_id`` and ``split``. The table has no identifier column.
    """
    frame = _join_split(prepared, split)
    train_mask = frame["split"].eq("train")
    if int(train_mask.sum()) == 0:
        raise ValueError("no training rows")

    raw_medians = {
        column: float(frame[column].median())
        for column in ("age", "bmi", "sugar_drinks")
    }
    train_medians = {}
    for column in ("age", "bmi", "sugar_drinks"):
        train_medians[column] = float(frame.loc[train_mask, column].median())
        frame[column] = frame[column].fillna(train_medians[column])

    modes = {}
    for column in (
        "any_exercise",
        "heavy_drinker",
        "smoker_status",
        "smokeless_tobacco",
        "ecigarette",
        "sex",
    ):
        modes[column] = _mode(frame.loc[train_mask, column])
        frame[column] = frame[column].fillna(modes[column])

    frame["sex"], rare_sex = collapse_rare_levels(frame["sex"], train_mask)
    frame["age_band"] = _band_code(frame["age"], AGE_BANDS)
    frame["bmi_band"] = _band_code(frame["bmi"], BMI_BANDS)
    frame["bmi_to_age"] = frame["bmi"] / frame["age"]

    encoded = pd.DataFrame({
        "age": frame["age"].astype(float),
        "bmi": frame["bmi"].astype(float),
        "sugar_drinks": frame["sugar_drinks"].astype(float),
        "any_exercise": frame["any_exercise"].map({True: 1.0, False: 0.0}).astype(float),
        "heavy_drinker": frame["heavy_drinker"].map({True: 1.0, False: 0.0}).astype(float),
        "smoker_ord": frame["smoker_status"].map(SMOKER_ORD).astype(float),
        "smokeless_ord": frame["smokeless_tobacco"].map(SMOKELESS_ORD).astype(float),
        "ecigarette_ord": frame["ecigarette"].map(ECIG_ORD).astype(float),
        "sex_male": (frame["sex"] == "male").astype(float),
        "age_band": frame["age_band"].astype(float),
        "bmi_band": frame["bmi_band"].astype(float),
        "bmi_to_age": frame["bmi_to_age"].astype(float),
    })
    if encoded.isna().any().any():
        raise ValueError("encoding left a missing value; a training mode was not in the maps")

    encoded, constant_columns = drop_constants(encoded, train_mask)
    encoded, duplicate_columns = drop_near_duplicates(encoded, train_mask)
    encoded["target"] = frame["target"].map(TARGET_CODE).astype(int)
    encoded["split"] = frame["split"].map(SPLIT_CODE).astype(int)
    if encoded["target"].isna().any() or encoded["split"].isna().any():
        raise ValueError("unmapped target or split")

    keys = pd.DataFrame({
        "row_id": frame["row_id"].to_numpy(),
        "split": frame["split"].to_numpy(),
        "target": frame["target"].to_numpy(),
    })
    report = {
        "random_state": SPLIT_SEED,
        "train_fraction": TRAIN_FRAC,
        "validation_fraction": VAL_FRAC,
        "test_fraction": TEST_FRAC,
        "train_medians": train_medians,
        "all_row_medians_before_impute": raw_medians,
        "train_modes": {key: _jsonable(value) for key, value in modes.items()},
        "rare_sex_levels": rare_sex,
        "dropped_constants": constant_columns,
        "dropped_duplicates": duplicate_columns,
        "age_labels": list(AGE_LABELS),
        "bmi_labels": list(BMI_LABELS),
        "target_codes": TARGET_CODE,
        "split_codes": SPLIT_CODE,
        "notes": {
            column: FEATURE_NOTES[column]
            for column in encoded.columns
            if column in FEATURE_NOTES
        },
    }
    return encoded, keys, report


def _jsonable(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return str(value)


def modeling_checks(table):
    """The four questions the week-4 handout asks before submission."""
    features = [column for column in table.columns if column not in ("target", "split")]
    train = table.loc[table["split"].eq(SPLIT_CODE["train"]), features]
    test = table.loc[table["split"].eq(SPLIT_CODE["test"]), features]
    numeric = all(np.issubdtype(table[column].dtype, np.number) for column in table.columns)
    return {
        "all_numeric": bool(numeric),
        "no_missing": int(table.isna().sum().sum()) == 0,
        "identifier_removed": "row_id" not in table.columns,
        "train_columns": int(train.shape[1]),
        "test_columns": int(test.shape[1]),
        "same_column_count": int(train.shape[1]) == int(test.shape[1]),
        "same_column_names": list(train.columns) == list(test.columns),
        "train_rows": int(table["split"].eq(SPLIT_CODE["train"]).sum()),
        "validation_rows": int(table["split"].eq(SPLIT_CODE["validation"]).sum()),
        "test_rows": int(table["split"].eq(SPLIT_CODE["test"]).sum()),
    }


def category_counts(prepared, split):
    """Training counts for the columns that could have been one-hot encoded."""
    frame = _join_split(prepared, split)
    train = frame.loc[frame["split"].eq("train")]
    rows = []
    for column in ("sex", "smoker_status", "smokeless_tobacco", "ecigarette", "any_exercise", "heavy_drinker"):
        counts = train[column].value_counts(dropna=False)
        for level, count in counts.items():
            rows.append({
                "column": column,
                "level": level,
                "train_n": int(count),
                "train_share": float(count / len(train)),
            })
    return pd.DataFrame(rows)


def load_prepared():
    if not THIN_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Thin table not found at {THIN_TABLE_PATH}. Run src/diabetes_risk/load.py first."
        )
    if not SPLIT_PATH.exists():
        raise FileNotFoundError(f"Frozen split not found at {SPLIT_PATH}.")
    thin = pd.read_parquet(THIN_TABLE_PATH)
    return prepare_table(thin), pd.read_csv(SPLIT_PATH)


def write_feature_table(table, keys):
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(FEATURES_PATH, index=False)
    keys.to_csv(KEYS_PATH, index=False)
    return FEATURES_PATH, KEYS_PATH


def main():
    prepared, split = load_prepared()
    table, keys, report = build_encoded(prepared, split)
    write_feature_table(table, keys)
    checks = modeling_checks(table)
    print(f"random_state={report['random_state']}  "
          f"train/validation/test={report['train_fraction']}/"
          f"{report['validation_fraction']}/{report['test_fraction']}")
    print(f"Saved {FEATURES_PATH}")
    print(f"Saved {KEYS_PATH} (row ids only, not a model input)")
    for name, passed in checks.items():
        print(f"{name}: {passed}")
    if report["dropped_constants"] or report["dropped_duplicates"]:
        print("dropped", report["dropped_constants"], report["dropped_duplicates"])
    else:
        print("dropped none: no constant column and no pair at or above "
              f"{NEAR_DUPLICATE} correlation on the training rows")
    return checks


if __name__ == "__main__":
    main()
