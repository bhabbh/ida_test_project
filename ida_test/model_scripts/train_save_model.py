"""Train Model and predict."""
import cloudpickle
import os
from ida_test.model_scripts.train_predict import read_data, encode_data, train_model
from pathlib import Path

# PATHS
root_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(root_dir,"ida_test/data/files")
model_path = os.path.join(data_dir,"model.pkl")
data_path = os.path.join(data_dir,"data.parquet")



def save_model(model):
    """saves model to model_path

    Args:
        model : ML model
    """
    with open(model_path, "wb") as f:
        cloudpickle.dump(model, f)        

# Load model
def load_model(model_path:Path):
    """saves model to model_path

    Args:
        model_path : path to the ML model
    Returns :
        model : ML model
    """
    with open(model_path, "rb") as f:
        model = cloudpickle.load(f)
    return model

if __name__ == "__main__":
    _, df,_ = read_data(data_path)
    df = encode_data(df)
    model, _, _ = train_model(df)
    save_model(model)
    print(f"model saved in {model_path}")