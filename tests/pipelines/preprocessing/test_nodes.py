import pandas as pd

from ai_project_titanic.pipelines.preprocessing.nodes import (
    clean_data,
    scale_data,
    split_data,
)


def test_clean_data_not_empty():
    df = pd.DataFrame({
        "Age": [22, None, 38],
        "Fare": [7.25, 71.28, None],
        "Embarked": ["S", None, "C"],
        "Cabin": [None, "C85", None],
    })

    cleaned = clean_data(df)

    assert cleaned["Age"].isna().sum() == 0
    assert cleaned["Embarked"].isna().sum() == 0
    assert "Cabin" not in cleaned.columns
    assert "CabinKnown" in cleaned.columns


def test_scale_data_standardized():
    df = pd.DataFrame({
        "Age": [10, 20, 30],
        "Fare": [7, 8, 9],
        "Embarked": ["S", "C", "Q"],
    })

    scaled = scale_data(df)

    age_mean = scaled["Age"].mean()
    fare_std = scaled["Fare"].std()

    assert abs(age_mean) < 1e-6
    assert abs(fare_std - 1) < 1e-6


def test_split_data_shapes():
    df = pd.DataFrame({"A": list(range(100))})

    train, val, test = split_data(df)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
