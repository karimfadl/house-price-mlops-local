"""Data quality tests. Skipped if CSV does not exist yet."""

import os
import pandas as pd
import pytest

DATA_PATH = "data/house_prices.csv"


@pytest.fixture(scope="module")
def df():
    if not os.path.exists(DATA_PATH):
        pytest.skip("Dataset not generated yet")
    return pd.read_csv(DATA_PATH)


def test_expected_columns(df):
    expected = {"sqft_living","bedrooms","bathrooms","house_age",
                "distance_city","garage","school_rating","price"}
    assert expected.issubset(set(df.columns))


def test_no_nulls(df):
    assert df.isnull().sum().sum() == 0


def test_price_positive(df):
    assert (df["price"] > 0).all()


def test_garage_binary(df):
    assert set(df["garage"].unique()).issubset({0, 1})


def test_school_rating_in_range(df):
    assert df["school_rating"].between(1.0, 10.0).all()


def test_minimum_rows(df):
    assert len(df) >= 1000
