"""front app"""

import streamlit as st
import pandas as pd
import requests
import streamlit as st

st.title('Sales Prediction')

# User inputs
store_options = ["1354820366865ba193741390bba9d17b", "c8c368f0311ea25b581cb3c704fe3a70"]
store_id = st.selectbox("Select a store:", store_options)
date = st.date_input("Prediction Date") # datetime.date YYYY-MM-DD
range_days = st.number_input("Range (days)", min_value=1, step=1) # int
sale_ids = st.text_input("Sale IDs (comma-separated)") # str



if st.button('Predict'):
    sale_ids = sale_ids.split(',') # list(str)
    for sale_id in sale_ids:
        response = requests.post('http://localhost:8000/predict', json={'store_id': str(store_id), 'sale_date': str(date), 'horizon': range_days, 'sale_id': sale_id.strip()})
        if response.status_code == 200:
            prediction_list, sale_name = response.json()
            if not prediction_list:
                st.write(f"{sale_name} : prediction unavailable for this store")
            else:
                prediction = pd.DataFrame(prediction_list)
                units_pred = prediction["XGBRegressor"].sum()
                units_pred_str = f"{int(units_pred)}-{int(units_pred)+1}"
                st.write(f"{sale_name} : {units_pred_str} units predicted")
                st.write()
        else:
            st.write(f"{sale_name} : Error in prediction")