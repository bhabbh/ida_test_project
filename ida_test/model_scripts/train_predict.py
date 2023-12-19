"""Train Model and predict."""
from datetime import date
from pathlib import Path
import os

import pandas as pd
from mlforecast import MLForecast
from sklearn.preprocessing import LabelEncoder
from window_ops.rolling import rolling_max, rolling_mean, rolling_min
from xgboost import XGBRegressor

# FILEPATHS
root_dir = os.path.abspath(os.getcwd())
data_dir = os.path.join(root_dir,"ida_test/data/files")
data_path = os.path.join(data_dir,"ida_test/data.parquet")


def read_data(fp: Path) -> pd.DataFrame:
    """read data from path.

    :fp: path to file holding data.
    """
    data = pd.read_parquet(fp)
    cat = data[["sale_id", "sale_name", "sub_family_name", "family_name"]].drop_duplicates()
    sale_id_to_name = data[["sale_id", "sale_name"]].drop_duplicates().reset_index(drop=True)
    data = data[["sale_date", "sale_id", "store_id", "sale_amount"]]
    return cat, data, sale_id_to_name


def encode_data(df: pd.DataFrame):
    """Encode data.

    Args:
        df: DataFrame holding sale_id, store_id to encode.
    """
    enc_store = LabelEncoder()
    enc_sale = LabelEncoder()
    df["unique_id"] = df["sale_id"] + "|" + df["store_id"]
    df["sale_id"] = enc_sale.fit_transform(df["sale_id"].values)
    df["store_id"] = enc_store.fit_transform(df["store_id"].values)
    return df


def train_model(df: pd.DataFrame):
    """
    Trains a machine learning model for time series forecasting using the provided DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the dataset with columns:
        - 'unique_id': Identifier for each observation.
        - 'sale_date': Date of the sale.
        - 'sale_amount': Amount of sale.
        - 'sale_id': Identifier for the sale.
        - 'store_id': Identifier for the store.

    Returns
    -------
    model : MLForecast
        The trained machine learning forecast model.

    train_df : pd.DataFrame
        The training subset of the provided DataFrame ('df') where 'sale_date' is before '2023-12-01'.

    val_df : pd.DataFrame
        The validation subset of the provided DataFrame ('df').

    Notes
    -----
    This function initializes a machine learning model for time series forecasting using XGBoost regressor
    and fits it on the provided dataset. It transforms features based on specified lags and date features.

    Example
    -------
    >>> trained_model, train_data, validation_data = train_model(dataset)
    """
    models = [XGBRegressor(random_state=0, n_estimators=100)]

    model = MLForecast(
        models=models,
        freq="D",
        lags=[1, 7, 14],
        lag_transforms={
            1: [(rolling_mean, 7), (rolling_max, 7), (rolling_min, 7)],
        },
        date_features=["dayofweek", "dayofyear", "week"],
        num_threads=6,
    )

    df = df[
        [
            "unique_id",
            "sale_date",
            "sale_amount",
            "sale_id",
            "store_id",
        ]
    ]
    train_df = df[df["sale_date"] < "2023-12-01"]
    val_df = df

    model.fit(
        train_df,
        id_col="unique_id",
        time_col="sale_date",
        target_col="sale_amount",
        static_features=["sale_id", "store_id"],
    )
    return model, train_df, val_df


def predict(
    model,
    horizon: int,
    X: pd.DataFrame,
    cat: pd.DataFrame,
    pred_date: date | str,
    store_id: str,
):
    pred = model.predict(h=horizon, new_df=X[X["sale_date"] < pred_date])
    pred[["sale_id", "store_id"]] = pred.unique_id.str.split("|", expand=True)
    store_pred = pred.loc[(pred["store_id"] == store_id) & (pred["sale_date"] >= pred_date), :]
    store_pred = store_pred.merge(cat, on="sale_id", how="left")
    return store_pred


if __name__ == "__main__":
    cat, df, _ = read_data(data_path)
    df = encode_data(df)
    model, train_df, val_df = train_model(df)
    store_pred = predict(model, 10, val_df, cat, pred_date="2023-12-10", store_id="c8c368f0311ea25b581cb3c704fe3a70")