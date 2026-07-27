"""
Streamlit app — Used Car Price Predictor
Loads the model artifacts produced by notebooks/used_car_price_prediction.ipynb
and lets the user get an instant price estimate for a used car.
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- Paths
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

# --------------------------------------------------------------------------- Page config
st.set_page_config(
    page_title="Used Car Price Predictor",
    layout="centered",
)

# --------------------------------------------------------------------------- Styling
st.markdown("""
<style>

/* ==============================
   General
============================== */

.stApp{
    background: linear-gradient(135deg,#08111f 0%,#10223b 45%,#173b63 100%);
    color:#f8fafc;
}

*{
    transition: all .25s ease;
}

/* ==============================
   Header
============================== */

h1{
    color:white !important;
    text-align:center;
    font-weight:800;
    letter-spacing:.4px;
    animation:fadeDown .6s ease;
}

h2,h3,h4,p,label,.stMarkdown{
    color:#E5EDF7 !important;
}

/* ==============================
   Inputs
============================== */

div[data-baseweb="input"]{
    background:#16273d;
    border-radius:12px;
    border:1px solid #36587d;
}

div[data-baseweb="input"]:hover{
    border-color:#58C4FF;
    box-shadow:0 0 12px rgba(88,196,255,.25);
}

input{
    color:white !important;
    font-size:15px !important;
    font-weight:600;
}

/* ==============================
   SelectBox
============================== */

div[data-baseweb="select"]>div{
    background:#16273d !important;
    color:white !important;
    border-radius:12px;
    border:1px solid #36587d;
}

div[data-baseweb="select"]:hover>div{
    border-color:#58C4FF;
    box-shadow:0 0 12px rgba(88,196,255,.25);
}

/* ==============================
   Button
============================== */

.stButton>button{

    width:100%;
    height:56px;

    border:none;
    border-radius:14px;

    font-size:18px;
    font-weight:700;

    color:white;

    background:linear-gradient(90deg,#ff8c42,#ffb703);

    box-shadow:0 8px 20px rgba(255,183,3,.25);

}

.stButton>button:hover{

    transform:translateY(-3px);

    box-shadow:0 12px 26px rgba(255,183,3,.45);

    background:linear-gradient(90deg,#ff9d5a,#ffc531);

}

.stButton>button:active{

    transform:scale(.98);

}

/* ==============================
   Divider
============================== */

hr{
    border:none;
    height:1px;
    background:#2E5275;
}

/* ==============================
   Expander
============================== */

details{

    background:#16273d;

    border-radius:16px;

    border:1px solid #315579;

    padding:10px;

}

details:hover{

    border-color:#58C4FF;

}

/* ==============================
   Metric
============================== */

div[data-testid="stMetric"]{

    background:#16273d;

    border-radius:16px;

    border:1px solid #36587d;

    padding:18px;

    box-shadow:0 5px 15px rgba(0,0,0,.25);

}

/* ==============================
   Price Card
============================== */

.price-card{

    background:linear-gradient(135deg,#1d3557,#27496d);

    border:1px solid rgba(255,255,255,.15);

    border-radius:20px;

    padding:35px;

    text-align:center;

    margin-top:10px;

    box-shadow:0 15px 35px rgba(0,0,0,.35);

    animation:fadeUp .5s ease;

}

.price-card:hover{

    transform:translateY(-4px);

    box-shadow:0 20px 45px rgba(0,0,0,.45);

}

.price-card h2{

    color:#ffb703 !important;

    font-size:3rem;

    margin:8px 0;

}

.price-card p{

    color:#cbd5e1 !important;

    font-size:18px;

}

/* ==============================
   Scrollbar
============================== */

::-webkit-scrollbar{

    width:10px;

}

::-webkit-scrollbar-thumb{

    background:#406A93;

    border-radius:20px;

}

::-webkit-scrollbar-thumb:hover{

    background:#58C4FF;

}

/* ==============================
   Animations
============================== */

@keyframes fadeUp{

    from{

        opacity:0;

        transform:translateY(25px);

    }

    to{

        opacity:1;

        transform:translateY(0);

    }

}

@keyframes fadeDown{

    from{

        opacity:0;

        transform:translateY(-20px);

    }

    to{

        opacity:1;

        transform:translateY(0);

    }

}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------- Load artifacts
@st.cache_resource
def load_artifacts():
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(COLUMNS_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, scaler, columns


model, scaler, feature_columns = load_artifacts()

# --------------------------------------------------------------------------- Header
st.title("Used Car Price Predictor")
st.write(
    "Estimate the fair resale value of a used car based on its age, mileage, "
    "engine specs, and ownership history — powered by a model trained on the "
    "CarDekho used-car dataset."
)

if model is None:
    st.error(
        "Model artifacts not found. Please run "
        "`notebooks/used_car_price_prediction.ipynb` first — it saves "
        "`best_model.pkl`, `scaler.pkl`, and `feature_columns.pkl` into the "
        "`models/` folder."
    )
    st.stop()

st.divider()

# --------------------------------------------------------------------------- Inputs
col1, col2 = st.columns(2)

with col1:
    year = st.number_input("Manufacturing year", min_value=1990, max_value=2026, value=2018, step=1)
    km_driven = st.number_input("KM driven", min_value=0, max_value=1_000_000, value=50_000, step=1000)
    mileage = st.number_input("Mileage (kmpl)", min_value=5.0, max_value=35.0, value=18.0, step=0.1)
    engine = st.number_input("Engine (CC)", min_value=600, max_value=5000, value=1200, step=50)

with col2:
    max_power = st.number_input("Max power (bhp)", min_value=30.0, max_value=400.0, value=85.0, step=1.0)
    seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10], index=2)
    fuel = st.selectbox("Fuel type", ["Diesel", "Petrol", "CNG", "LPG"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
owner = st.selectbox(
    "Ownership history",
    ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"],
)

st.divider()

# --------------------------------------------------------------------------- Predict
def build_feature_row():
    age = 2026 - year
    row = {col: 0 for col in feature_columns}

    row["km_driven"] = km_driven
    row["mileage"] = mileage
    row["engine"] = engine
    row["max_power"] = max_power
    row["seats"] = seats
    row["age"] = age

    if f"fuel_{fuel}" in row:
        row[f"fuel_{fuel}"] = 1
    if f"seller_type_{seller_type}" in row:
        row[f"seller_type_{seller_type}"] = 1
    if transmission == "Manual" and "transmission_Manual" in row:
        row["transmission_Manual"] = 1
    if f"owner_{owner}" in row:
        row[f"owner_{owner}"] = 1

    return pd.DataFrame([row])[feature_columns]


if st.button("Predict price", use_container_width=True):
    X_input = build_feature_row()
    X_scaled = scaler.transform(X_input)
    prediction = model.predict(X_scaled)[0]
    prediction = max(0, prediction)

    st.markdown(
        f"""
        <div class="price-card">
            <p>Estimated selling price</p>
            <h2>₹ {prediction:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "This is an estimate based on historical data trends and should be used "
        "as a reference point, not a formal valuation."
    )

st.divider()
with st.expander("About this app"):
    st.write(
        "This app uses the best-performing regression model identified in the "
        "training notebook (see `notebooks/used_car_price_prediction.ipynb`), "
        "trained on cleaned & engineered features from the CarDekho used-car "
        "dataset. Currency shown is Indian Rupees (₹), matching the source dataset."
    )
