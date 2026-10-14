import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from diabetes_risk.describe import (
    build_clean_table,
    height_cm,
    weight_kg,
    weighted_distribution,
    weighted_mean,
    weighted_share,
)


def test_height_reads_feet_inches_and_metric():
    heights = height_cm(pd.Series([507, 600, 9170, 7777, 9999, None]))
    assert heights[0] == pytest.approx(67 * 2.54)
    assert heights[1] == pytest.approx(72 * 2.54)
    assert heights[2] == 170
    assert heights[3:].isna().all()


def test_weight_reads_pounds_and_metric():
    weights = weight_kg(pd.Series([140, 9072, 7777, 9999, None]))
    assert weights[0] == pytest.approx(140 * 0.45359237)
    assert weights[1] == 72
    assert weights[2:].isna().all()


def test_height_and_weight_reproduce_bmi():
    # 5 ft 7 in, 140 lb is BMI 21.93 (_BMI5 = 2193 in the thin table).
    h = height_cm(pd.Series([507]))[0] / 100
    w = weight_kg(pd.Series([140]))[0]
    assert w / h**2 == pytest.approx(21.93, abs=0.01)


def test_weighted_mean_skips_missing():
    values = pd.Series([10.0, 20.0, np.nan])
    weights = pd.Series([1.0, 3.0, 100.0])
    assert weighted_mean(values, weights) == pytest.approx(17.5)


def test_weighted_share_skips_missing():
    flags = pd.Series([True, False, np.nan], dtype="object")
    weights = pd.Series([1.0, 3.0, 100.0])
    assert weighted_share(flags, weights) == pytest.approx(0.25)


def test_weighted_distribution_sums_to_one():
    groups = pd.Series(["a", "b", "b", None])
    weights = pd.Series([1.0, 1.0, 2.0, 50.0])
    shares = weighted_distribution(groups, weights)
    assert shares["a"] == pytest.approx(0.25)
    assert shares.sum() == pytest.approx(1.0)


def test_clean_table_adds_descriptive_columns():
    rows = pd.DataFrame(
        {
            "row_id": [1], "DIABETE4": [1], "_AGE80": [45], "_BMI5": [2193],
            "EXERANY2": [1], "_SMOKER3": [4], "USENOW3": [3], "ECIGNOW3": [1],
            "_RFDRHV9": [1], "SEXVAR": [1], "SSBSUGR2": [888],
            "HEIGHT3": [507], "WEIGHT2": [140], "_LLCPWT": [18.5],
        }
    )
    clean = build_clean_table(rows)
    assert clean.loc[0, "target"] == "diabetes"
    assert clean.loc[0, "height_cm"] == 170.2
    assert clean.loc[0, "weight_kg"] == 63.5
    assert clean.loc[0, "survey_weight"] == 18.5
