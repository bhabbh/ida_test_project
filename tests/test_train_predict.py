# run pytest from the root directory

import pytest
import pandas as pd
import os
from pathlib import Path
import sys
from mlforecast import MLForecast


sys.path.append(str(Path(__file__).parent.parent))
from ida_test.model_scripts.train_predict import read_data, encode_data, train_model, predict

# PATHS
root_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(root_dir,"ida_test/data/files")
test_dir = os.path.join(root_dir,"tests")
DATA_FILE_PATH = os.path.join(data_dir,"data.parquet")

# Read the data 
@pytest.fixture(scope="module")
def real_data():
    return read_data(DATA_FILE_PATH)

# Test read_data function
def test_read_data():
    cat, data, sale_id_to_name = read_data(DATA_FILE_PATH)

    # Test DataFrame structure
    assert isinstance(cat, pd.DataFrame)
    assert isinstance(data, pd.DataFrame)
    assert isinstance(sale_id_to_name, pd.DataFrame)

    # Test DataFrame columns
    expected_cat_columns = ["sale_id", "sale_name", "sub_family_name", "family_name"]
    expected_data_columns = ["sale_date", "sale_id", "store_id", "sale_amount"]
    expected_sale_id_to_name_columns = ["sale_id", "sale_name"]

    assert all(column in cat.columns for column in expected_cat_columns)
    assert all(column in data.columns for column in expected_data_columns)
    assert all(column in sale_id_to_name.columns for column in expected_sale_id_to_name_columns)

    # Test Data Integrity
    assert cat.duplicated().sum() == 0  # No duplicates in cat
    assert sale_id_to_name.duplicated().sum() == 0  # No duplicates in sale_id_to_name

# Test encode_data function
def test_encode_data(real_data):
    _, data, _ = real_data
    original_row_count = data.shape[0]
    encoded_data = encode_data(data.copy())

    # Check DataFrame structure
    assert 'unique_id' in encoded_data.columns
    assert 'sale_id' in encoded_data.columns
    assert 'store_id' in encoded_data.columns

    # Test data types
    assert pd.api.types.is_integer_dtype(encoded_data['sale_id'])
    assert pd.api.types.is_integer_dtype(encoded_data['store_id'])

    # Unique Identifier Test
    for _, row in data.sample(n=5).iterrows():
        encoded_row = encoded_data[encoded_data['unique_id'] == row['sale_id'] + '|' + row['store_id']].iloc[0]
        assert row['sale_id'] + '|' + row['store_id'] == encoded_row['unique_id']

    # Non-Negative Encoding
    assert encoded_data['sale_id'].min() >= 0
    assert encoded_data['store_id'].min() >= 0

    # Preserve Row Count
    assert original_row_count == encoded_data.shape[0]

    # No Data Loss
    assert not encoded_data.isnull().any().any()  # No null values in any column


# Test train_model function
def test_train_model(real_data):
    _, df, _ = real_data
    df = encode_data(df)
    model, train_df, val_df = train_model(df)

    # Test Model Training
    assert model is not None
    assert isinstance(model, MLForecast)

    # Test Data Splitting
    assert not train_df.empty
    assert not val_df.empty
    assert all(train_df['sale_date'] < pd.Timestamp("2023-12-01"))

    # Test DataFrame Structure
    expected_columns = ["unique_id", "sale_date", "sale_amount", "sale_id", "store_id"]
    assert all(column in train_df.columns for column in expected_columns)
    assert all(column in val_df.columns for column in expected_columns)


# Test predict function
def test_predict():
    horizon = 10
    pred_date = "2023-12-10"
    store_id1 = "c8c368f0311ea25b581cb3c704fe3a70"
    store_id2 = "1354820366865ba193741390bba9d17b"

    cat, df, _ = read_data(DATA_FILE_PATH)
    df = encode_data(df)
    model, _,  val_df = train_model(df)
    predictions1 = predict(model, horizon, val_df, cat, pred_date, store_id=store_id1)
    predictions2 = predict(model, horizon, val_df, cat, pred_date, store_id=store_id2)

    # Test Prediction Output
    assert not predictions1.empty
    assert all(predictions1['sale_date'] >= pd.Timestamp(pred_date))
    assert all(predictions1['store_id'] == store_id1)
    assert not predictions2.empty
    assert all(predictions2['sale_date'] >= pd.Timestamp(pred_date))
    assert all(predictions2['store_id'] == store_id2)

if __name__ == "__main__":
    pytest.main()


    