"""Week-4 feature table: train-only fills, numeric output, no identifier."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diabetes_risk.features import (
    build_encoded,
    drop_constants,
    drop_near_duplicates,
    modeling_checks,
)


def _frame(n_train=8):
    n = n_train + 4
    row_id = list(range(1, n + 1))
    split = ["train"] * n_train + ["validation", "validation", "test", "test"]
    target = (["no_diabetes", "diabetes", "prediabetes", "no_diabetes"] * n)[:n]
    age = [30, 40, 50, 60, 35, 45, 55, 70][:n_train] + [20, 80, 33, 44]
    bmi = [22.0, np.nan, 27.0, 24.0, 26.0, 23.0, 28.0, 21.0][:n_train] + [90.0, 19.0, 25.0, 30.0]
    sex = (["female", "male"] * n)[:n]
    smoker = (["never", "former", "current_some_days", "current_every_day"] * n)[:n]
    prepared = pd.DataFrame({
        "row_id": row_id,
        "target": target,
        "age": age,
        "bmi": bmi,
        "any_exercise": [True, False] * (n // 2) + [True] * (n % 2),
        "smoker_status": smoker,
        "smokeless_tobacco": (["not_at_all", "some_days", "every_day", "not_at_all"] * n)[:n],
        "ecigarette": (["never", "former", "some_days", "every_day"] * n)[:n],
        "heavy_drinker": [False, True] * (n // 2) + [False] * (n % 2),
        "sex": sex,
        "sugar_drinks": np.linspace(0, 3, n),
    })
    split_frame = pd.DataFrame({"row_id": row_id, "split": split})
    return prepared, split_frame


def test_imputation_ignores_validation_and_test_extremes():
    prepared, split = _frame()
    table, keys, report = build_encoded(prepared, split)
    # Training BMI values that are present: 22, 27, 24, 26, 23, 28, 21. Median 24.
    # The missing training BMI must become that median, not a median that includes 90.
    missing_row = keys.index[keys["row_id"].eq(2)][0]
    assert table.loc[missing_row, "bmi"] == pytest.approx(24.0)
    assert report["train_medians"]["bmi"] == pytest.approx(24.0)
    assert report["all_row_medians_before_impute"]["bmi"] != pytest.approx(24.0)


def test_table_is_numeric_complete_and_has_no_identifier():
    prepared, split = _frame()
    table, keys, _report = build_encoded(prepared, split)
    checks = modeling_checks(table)
    assert checks["all_numeric"]
    assert checks["no_missing"]
    assert checks["identifier_removed"]
    assert checks["same_column_count"]
    assert checks["same_column_names"]
    assert "row_id" not in table.columns
    assert keys["row_id"].is_unique
    assert len(keys) == len(table)


def test_second_call_matches_the_first():
    prepared, split = _frame()
    first, first_keys, _report = build_encoded(prepared, split)
    second, second_keys, _report = build_encoded(prepared, split)
    pd.testing.assert_frame_equal(first, second)
    pd.testing.assert_frame_equal(first_keys, second_keys)


def test_constant_and_duplicate_columns_are_dropped():
    frame = pd.DataFrame({
        "age": [1.0, 2.0, 3.0, 4.0],
        "flat": [1.0, 1.0, 1.0, 1.0],
        "age_copy": [1.0, 2.0, 3.0, 4.0],
    })
    train = pd.Series([True, True, True, False])
    kept, constants = drop_constants(frame, train)
    assert constants == ["flat"]
    kept, duplicates = drop_near_duplicates(kept, train)
    assert duplicates[0]["column"] == "age_copy"
    assert duplicates[0]["duplicate_of"] == "age"
    assert list(kept.columns) == ["age"]
