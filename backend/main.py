# import os
# import sys
# import types
# import pandas as pd
# import joblib

# # EXPLICITLY IMPORT sklearn AND HistGradientBoostingRegressor BEFORE LOADING JOBLIB!
# import sklearn
# from sklearn.ensemble import HistGradientBoostingRegressor

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware


# # 1. Initialize the app
# app = FastAPI(title="Medicine Price & Alternative Finder API")

# # Enable CORS so your frontend (HTML/JS or Streamlit) can talk to this API
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# def _install_hist_gradient_boosting_pickle_compat():
#     """Allow older sklearn pickles to load on newer sklearn versions."""
#     try:
#         import sklearn.ensemble._hist_gradient_boosting.gradient_boosting as gb
#     except Exception:
#         return

#     try:
#         import importlib
#         importlib.import_module("sklearn.ensemble._hist_gradient_boosting._loss")
#         return
#     except ModuleNotFoundError:
#         pass

#     compat_module = types.ModuleType("sklearn.ensemble._hist_gradient_boosting._loss")
#     for attr_name in [
#         "BaseLoss",
#         "HalfBinomialLoss",
#         "HalfGammaLoss",
#         "HalfMultinomialLoss",
#         "HalfPoissonLoss",
#         "PinballLoss",
#         "CyHalfSquaredError",
#     ]:
#         if hasattr(gb, attr_name):
#             setattr(compat_module, attr_name, getattr(gb, attr_name))

#     if not hasattr(compat_module, "CyHalfSquaredError"):
#         try:
#             from sklearn._loss.loss import HalfSquaredError
#             compat_module.CyHalfSquaredError = HalfSquaredError
#         except Exception:
#             pass

#     sys.modules["sklearn.ensemble._hist_gradient_boosting._loss"] = compat_module
#     sys.modules["_loss"] = compat_module


# _install_hist_gradient_boosting_pickle_compat()

# # 2. Load the data and models safely (Happens once at server startup)
# DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/cleaned_medicines.csv')
# MODEL_PATH = os.path.join(os.path.dirname(__file__), '../model/rf_model.joblib')
# TRF_PATH = os.path.join(os.path.dirname(__file__), '../model/preprocessor.joblib')
# PT_PATH = os.path.join(os.path.dirname(__file__), '../model/power_transformer.joblib')

# df = pd.read_csv(DATA_PATH)
# model = joblib.load(MODEL_PATH)
# trf = joblib.load(TRF_PATH)
# pt = joblib.load(PT_PATH)

# # Ensure unit_price column exists for fair comparisons
# df['pack_size_valid'] = df['pack_size'].apply(lambda x: x if pd.notna(x) and x > 0 else 1.0)
# df['unit_price'] = df['price_inr'] / df['pack_size_valid']


# # Root Endpoint
# @app.get("/")
# def home():
#     return {"message": "Welcome to the Medicine Price & Alternative Finder API!"}


# # 3. Endpoint 1: Get Cheaper Alternatives
# @app.get("/alternatives")
# def get_alternatives(medicine_name: str):
#     # Case-insensitive search for medicine brand
#     med_row = df[df['brand_name'].str.lower() == medicine_name.lower()]
    
#     if med_row.empty:
#         raise HTTPException(
#             status_code=404, 
#             detail=f"Medicine '{medicine_name}' not found. Please check spelling."
#         )
    
#     selected_med = med_row.iloc[0]
    
#     # FIX 1: Match on cleaned_composition (NOT primary_ingredient)
#     target_comp = selected_med['cleaned_composition']
#     current_unit_price = selected_med['unit_price']
    
#     # FIX 3: Filter by cleaned_composition AND unit_price
#     cheaper_meds = df[
#         (df['cleaned_composition'] == target_comp) & 
#         (df['unit_price'] < current_unit_price) &
#         (df['brand_name'].str.lower() != medicine_name.lower())
#     ]
    
#     if cheaper_meds.empty:
#         return {
#             "medicine_name": selected_med['brand_name'],
#             "composition": target_comp,
#             "alternatives": [],
#             "message": "No cheaper alternatives found for this composition."
#         }
    
#     # Sort by cheapest unit price and pick top 5
#     top_5 = cheaper_meds.sort_values(by='unit_price').head(5)
    
#     result = top_5[[
#         'brand_name', 
#         'manufacturer', 
#         'price_inr', 
#         'pack_size', 
#         'pack_unit'
#     ]].to_dict(orient='records')
    
#     return {
#         "searched_medicine": selected_med['brand_name'],
#         "composition": target_comp,
#         "alternatives": result
#     }


# # 4. Endpoint 2: Predict Fair Price
# @app.get("/predict-price")
# def predict_price(medicine_name: str):
#     med_row = df[df['brand_name'].str.lower() == medicine_name.lower()]
    
#     if med_row.empty:
#         raise HTTPException(
#             status_code=404, 
#             detail=f"Medicine '{medicine_name}' not found."
#         )
    
#     # FIX 2: Explicitly select ONLY the exact features used during model training
#     feature_columns = [
#         'manufacturer', 
#         'is_discontinued', 
#         'dosage_form', 
#         'pack_size', 
#         'pack_unit', 
#         'num_active_ingredients', 
#         'cleaned_composition'
#     ]
    
#     input_data = med_row[feature_columns].copy()
    
#     # Fill NAs exactly as done during training
#     input_data['pack_size'] = input_data['pack_size'].fillna(-1)
#     input_data['pack_unit'] = input_data['pack_unit'].fillna('Unknown')
#     input_data['is_discontinued'] = input_data['is_discontinued'].astype(int)
    
#     try:
#         # Transform input using preprocessor pipeline
#         transformed_data = trf.transform(input_data)
        
#         # Predict transformed price using Random Forest
#         pred_transformed = model.predict(transformed_data)
        
#         # Inverse transform to get price in Rupees
#         fair_price = pt.inverse_transform(pred_transformed.reshape(-1, 1))[0][0]
#         actual_price = float(med_row.iloc[0]['price_inr'])
        
#         return {
#             "medicine_name": med_row.iloc[0]['brand_name'],
#             "composition": med_row.iloc[0]['cleaned_composition'],
#             "actual_price": round(actual_price, 2),
#             "predicted_fair_price": round(float(fair_price), 2)
#         }
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


import os
import pandas as pd
import joblib

# Explicitly import sklearn & HistGradientBoostingRegressor
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Medicine Price & Alternative Finder API")

# Enable CORS for Streamlit frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------
# ABSOLUTE FILE PATH RESOLUTION (Guaranteed to work on Render Cloud)
# -------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) # .../backend
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)              # .../project root

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "cleaned_medicines.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "rf_model.joblib")
TRF_PATH = os.path.join(PROJECT_ROOT, "model", "preprocessor.joblib")
PT_PATH = os.path.join(PROJECT_ROOT, "model", "power_transformer.joblib")

# Load dataset and ML model files safely
df = pd.read_csv(DATA_PATH)
model = joblib.load(MODEL_PATH)
trf = joblib.load(TRF_PATH)
pt = joblib.load(PT_PATH)

# Calculate unit_price for fair comparisons
df['pack_size_valid'] = df['pack_size'].apply(lambda x: x if pd.notna(x) and x > 0 else 1.0)
df['unit_price'] = df['price_inr'] / df['pack_size_valid']


@app.get("/")
def home():
    return {"message": "Medicine Price API is live and running!"}


@app.get("/alternatives")
def get_alternatives(medicine_name: str):
    med_row = df[df['brand_name'].str.lower() == medicine_name.lower()]
    
    if med_row.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Medicine '{medicine_name}' not found."
        )
    
    selected_med = med_row.iloc[0]
    target_comp = selected_med['cleaned_composition']
    current_unit_price = selected_med['unit_price']
    
    cheaper_meds = df[
        (df['cleaned_composition'] == target_comp) & 
        (df['unit_price'] < current_unit_price) &
        (df['brand_name'].str.lower() != medicine_name.lower())
    ]
    
    if cheaper_meds.empty:
        return {
            "searched_medicine": selected_med['brand_name'],
            "composition": target_comp,
            "alternatives": [],
            "message": "No cheaper alternatives found."
        }
    
    top_5 = cheaper_meds.sort_values(by='unit_price').head(5)
    
    result = top_5[[
        'brand_name', 
        'manufacturer', 
        'price_inr', 
        'pack_size', 
        'pack_unit'
    ]].to_dict(orient='records')
    
    return {
        "searched_medicine": selected_med['brand_name'],
        "composition": target_comp,
        "alternatives": result
    }


@app.get("/predict-price")
def predict_price(medicine_name: str):
    med_row = df[df['brand_name'].str.lower() == medicine_name.lower()]
    
    if med_row.empty:
        raise HTTPException(
            status_code=404, 
            detail=f"Medicine '{medicine_name}' not found."
        )
    
    feature_columns = [
        'manufacturer', 
        'is_discontinued', 
        'dosage_form', 
        'pack_size', 
        'pack_unit', 
        'num_active_ingredients', 
        'cleaned_composition'
    ]
    
    input_data = med_row[feature_columns].iloc[[0]].copy()
    
    input_data['pack_size'] = input_data['pack_size'].fillna(-1)
    input_data['pack_unit'] = input_data['pack_unit'].fillna('Unknown')
    input_data['is_discontinued'] = input_data['is_discontinued'].astype(int)
    
    try:
        transformed_data = trf.transform(input_data)
        pred_transformed = model.predict(transformed_data)
        fair_price = pt.inverse_transform(pred_transformed.reshape(-1, 1))[0][0]
        actual_price = float(med_row.iloc[0]['price_inr'])
        
        return {
            "medicine_name": med_row.iloc[0]['brand_name'],
            "composition": med_row.iloc[0]['cleaned_composition'],
            "actual_price": round(actual_price, 2),
            "predicted_fair_price": round(float(fair_price), 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}") 
    