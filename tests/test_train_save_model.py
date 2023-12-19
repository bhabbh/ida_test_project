import sys
import os
from pathlib import Path
import pytest
from mlforecast import MLForecast
sys.path.append(str(Path(__file__).parent.parent))
from ida_test.model_scripts.train_predict import encode_data, train_model, read_data
from ida_test.model_scripts.train_save_model import save_model, load_model  # Adjust the import based on your project structure

# expected model_path
root_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(root_dir,"ida_test/data/files")
model_path = Path(os.path.join(data_dir,"model.pkl"))
DATA_FILE_PATH = os.path.join(data_dir,"data.parquet")

# Fixture for a toy model
@pytest.fixture
def toy_model():
    _, df,_ = read_data(DATA_FILE_PATH)
    df = encode_data(df)
    model, _, _ = train_model(df)
    return model

# Test Save Model Function
def test_save_model(toy_model):
    save_model(toy_model)
    # Check file exists and is not empty
    assert model_path.exists()
    assert model_path.stat().st_size > 0

# Test Load Model Function
def test_load_model(toy_model):
    save_model(toy_model)
    loaded_model = load_model(model_path)
    # Check loaded model is of correct type
    assert isinstance(loaded_model, MLForecast)

# Test Round-Trip Save and Load
def test_round_trip_save_load(toy_model):
    save_model(toy_model)
    loaded_model = load_model(model_path)
    assert isinstance(loaded_model, type(toy_model))
    # Check Model Parameters
    if hasattr(toy_model, 'get_params') and callable(getattr(toy_model, 'get_params')):
        original_params = toy_model.get_params()
        loaded_params = loaded_model.get_params()
        assert original_params == loaded_params