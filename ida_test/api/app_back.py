"""back app"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ida_test.model_scripts.train_predict import read_data, encode_data, predict
from ida_test.model_scripts.train_save_model import load_model
from ida_test.data.features import add_cyclical_calendar_features
import os

# PATHS
root_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(root_dir,"ida_test/data/files")
model_path = os.path.join(data_dir,"model.pkl")
data_path = os.path.join(data_dir,"data.parquet")

# Define request model
class PredictionRequest(BaseModel):
    store_id : str
    sale_date: str
    sale_id: str
    horizon: int

# Initialize the FastAPI app
app = FastAPI()
# Load model
model = load_model(model_path)
# Load and preprocess the data
cat, data, sale_id_to_name = read_data(data_path)
df = add_cyclical_calendar_features(data, features=["day", "week"])
df = encode_data(df)
df = df[
    [
        "unique_id",
        "sale_date",
        "sale_amount",
        "sale_id",
        "store_id",
    ]
]

# Endpoint for predictions
@app.post("/predict")
async def make_prediction(request: PredictionRequest):
    """_summary_

    Args:
        request (PredictionRequest)
    Returns:
        sale_pred_list (List) : list of predictions
        sale_name (str) : sale name for the select product
    """
    try:
        # Generate predictions
        store_pred = predict(model, request.horizon, df, cat, request.sale_date, store_id=request.store_id)
        sale_pred = store_pred[store_pred["sale_id"] == request.sale_id]
        sale_pred_list = sale_pred.to_dict(orient='records')
        sale_name = sale_id_to_name[sale_id_to_name["sale_id"] == request.sale_id]["sale_name"].values[0]
        # Return the predictions
        return sale_pred_list, sale_name
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))