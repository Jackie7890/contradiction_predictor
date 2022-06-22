import numpy as np
import pandas as pd
import pytest
from contradiction_app.train_and_predict import clean_data, engineer_features, impute_median, load_and_prep_data
from mock import MagicMock, patch


@pytest.fixture
def sample_dataset():
    return pd.DataFrame({
        'numeric_1': [1.0, 2.0, 3.0],
        'numeric_2': [5, None, 3],
        'cat_1': ['a', 'b', 'c'],
        'cat_2': ['x', None, 'z'],
        'lot_size_square_feet': [0, 5000, 7500],
        'sold_date': ['2020-01-01', '2019-10-23', '2021-02-01'],
        'total_assessed_value': [10000, 20000, 30000],
        'assessed_land_value': [2500, 2500, 2500],
        'assessed_improvement_value': [7500, 17500, 27500],
        'building_area': [2000, 2250, 1450],
    })


def test_impute_median_correct_imputation(sample_dataset):
    missing_mask = sample_dataset.numeric_2.isna()
    # ensure missing values in column
    assert missing_mask.sum() == 1
    median = sample_dataset.numeric_2.median()

    impute_median(sample_dataset, [c for c in sample_dataset.columns if c.startswith("numeric_")])

    # ensure no longer missing
    assert sample_dataset.numeric_2.isna().sum() == 0

    # ensure median was used for imputation
    assert sample_dataset.numeric_2[missing_mask].values[0] == median


def test_impute_median_raise_categoricals(sample_dataset):
    with pytest.raises(ValueError, match="Feature cat_1 cannot be cast to numeric."):
        impute_median(sample_dataset, [c for c in sample_dataset.columns if c.startswith("cat_")])


def test_clean_data(sample_dataset):
    # ensure there is a 0 value in `lot_size_square_feet`
    zero_mask = sample_dataset.lot_size_square_feet == 0
    assert zero_mask.sum() == 1

    clean_data(sample_dataset)

    # ensure no one more 0 values
    assert (sample_dataset.lot_size_square_feet == 0).sum() == 0
    # ensure the previously 0 value is now missing
    assert sample_dataset.lot_size_square_feet[zero_mask].isna()[0]


@pytest.mark.parametrize("predict", [True, False])
def test_engineer_features(predict, sample_dataset):
    feat = engineer_features(sample_dataset, predict)
    # ensure FE are in dataset now
    for feature in feat:
        assert feature in sample_dataset.columns

    # ensure `days_since_sold` is always 0 when predicting but not when training
    if predict:
        assert all(sample_dataset.days_since_sold == 0)
    else:
        assert all(sample_dataset.days_since_sold.values == np.asarray([-397, -467, 0]))


@patch("homebound_mle.train_and_predict.pd.read_csv", return_value=MagicMock())
@patch("homebound_mle.train_and_predict.clean_data", return_value=MagicMock())
@patch("homebound_mle.train_and_predict.impute_median", return_value=MagicMock())
@patch("homebound_mle.train_and_predict.engineer_features", return_value=MagicMock())
def test_load_and_prep_data(
    mock_engineer_features, mock_impute_median, mock_clean_data, mock_read_csv, sample_dataset
):
    mock_read_csv.return_value = sample_dataset
    mock_engineer_features.return_value = []
    load_and_prep_data(numeric_feat=['numeric_1', 'numeric_2'], categorical_feat=['cat_1', 'cat_2'])
    # assert each method was called once
    assert mock_read_csv.call_count == 1
    assert mock_clean_data.call_count == 1
    assert mock_impute_median.call_count == 1
    assert mock_engineer_features.call_count == 1


