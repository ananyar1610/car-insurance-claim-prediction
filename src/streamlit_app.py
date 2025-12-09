# streamlit_app.py  (you can rename to streamlit.py)
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from feature_engineering import add_interactions
from utils import BOOL_COLS


# Page configuration

st.set_page_config(
    page_title="Car Insurance Claim Prediction",
    page_icon="🚗",
    layout="wide",
)

st.title("Car Insurance Claim Prediction")
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Home", "Data Overview", "Prediction"])



# Helper: load model & encoders

@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


@st.cache_resource
def load_encoders(encoders_path: str):
    return joblib.load(encoders_path)


MODEL_PATH = "models/best_model_XGBoost.pkl"  
ENCODERS_PATH = "models/all_encoders.pkl"


if page == "Home":
    st.subheader("Introduction")
    st.markdown(
        """
        This app predicts whether a **car insurance policy** is likely to result in a claim,
        based on vehicle details and policyholder information.
        """
    )

elif page == "Data Overview":
    st.subheader("Data Overview")
    st.markdown("Data overview section is under construction. You can add EDA here later.")

elif page == "Prediction":
    st.subheader("Prediction")

    # Load model & encoders
    model = load_model(MODEL_PATH)
    encoders = load_encoders(ENCODERS_PATH)

    st.markdown("### Categorical Fields")

    fuel_type = st.selectbox("Fuel Type", encoders["fuel_type"].classes_)
    rear_brakes_type = st.selectbox(
        "Rear Brakes Type", encoders["rear_brakes_type"].classes_
    )
    transmission_type = st.selectbox(
        "Transmission Type", encoders["transmission_type"].classes_
    )
    steering_type = st.selectbox("Steering Type", encoders["steering_type"].classes_)
    model_sel = st.selectbox("Model", encoders["model"].classes_)
    engine_type = st.selectbox("Engine Type", encoders["engine_type"].classes_)
    segment = st.selectbox("Segment", encoders["segment"].classes_)
    area_cluster = st.selectbox("Area Cluster", encoders["area_cluster"].classes_)

    st.markdown("### Numeric Fields")
    numeric_cols = [
        "policy_tenure",
        "age_of_car",
        "age_of_policyholder",
        "population_density",
        "airbags",
        "displacement",
        "cylinder",
        "turning_radius",
        "length",
        "width",
        "height",
        "gross_weight",
        "ncap_rating",
        "gear_box",
        "make",
    ]

    user_input = {}

    for col in numeric_cols:
        user_input[col] = st.number_input(
            col.replace("_", " ").title(), min_value=0.0, value=0.0
        )

    st.markdown("### Torque & Power")
    user_input["max_power_value"] = st.number_input(
        "Max Power Value (bhp)", min_value=0.0, value=60.0
    )
    user_input["max_torque_value"] = st.number_input(
        "Max Torque Value (Nm)", min_value=0.0, value=60.0
    )
    user_input["max_power_rpm"] = st.number_input(
        "Max Power RPM", min_value=0.0, value=3500.0
    )
    user_input["max_torque_rpm"] = st.number_input(
        "Max Torque RPM", min_value=0.0, value=3500.0
    )

    st.markdown("### Boolean Fields")
    for col in BOOL_COLS:
        val = st.selectbox(col.replace("_", " ").title(), ["Yes", "No"])
        user_input[col] = 1 if val == "Yes" else 0

    # Encode categorical fields
    user_input["fuel_type"] = encoders["fuel_type"].transform([fuel_type])[0]
    user_input["rear_brakes_type"] = encoders["rear_brakes_type"].transform(
        [rear_brakes_type]
    )[0]
    user_input["transmission_type"] = encoders["transmission_type"].transform(
        [transmission_type]
    )[0]
    user_input["steering_type"] = encoders["steering_type"].transform([steering_type])[
        0
    ]
    user_input["model"] = encoders["model"].transform([model_sel])[0]
    user_input["engine_type"] = encoders["engine_type"].transform([engine_type])[0]
    user_input["segment"] = encoders["segment"].transform([segment])[0]
    user_input["area_cluster"] = encoders["area_cluster"].transform([area_cluster])[0]

    # Build single-row DataFrame
    input_df = pd.DataFrame([user_input])

    # Add interaction features using the same logic as training
    input_df = add_interactions(input_df)

    # Enforce dtypes similar to training (you can tweak these)
    dtypes = {
        "policy_tenure": "float64",
        "age_of_car": "float64",
        "age_of_policyholder": "float64",
        "area_cluster": "int64",
        "population_density": "int64",
        "make": "int64",
        "segment": "int64",
        "model": "int64",
        "fuel_type": "int64",
        "engine_type": "int64",
        "airbags": "int64",
        "is_esc": "int64",
        "is_adjustable_steering": "int64",
        "is_tpms": "int64",
        "is_parking_sensors": "int64",
        "is_parking_camera": "int64",
        "rear_brakes_type": "int64",
        "displacement": "int64",
        "cylinder": "int64",
        "transmission_type": "int64",
        "gear_box": "int64",
        "steering_type": "int64",
        "turning_radius": "float64",
        "length": "int64",
        "width": "int64",
        "height": "int64",
        "gross_weight": "int64",
        "is_front_fog_lights": "int64",
        "is_rear_window_wiper": "int64",
        "is_rear_window_washer": "int64",
        "is_rear_window_defogger": "int64",
        "is_brake_assist": "int64",
        "is_power_door_locks": "int64",
        "is_central_locking": "int64",
        "is_power_steering": "int64",
        "is_driver_seat_height_adjustable": "int64",
        "is_day_night_rear_view_mirror": "int64",
        "is_ecw": "int64",
        "is_speed_alert": "int64",
        "ncap_rating": "int64",
        "max_power_value": "float64",
        "max_power_rpm": "float64",
        "max_torque_value": "float64",
        "max_torque_rpm": "float64",
        "power_to_weight": "float64",
        "torque_to_weight": "float64",
        "car_age_ratio": "float64",
        "engine_efficiency": "float64",
    }

    for col, col_type in dtypes.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(col_type)

    # Align feature order to model
    if hasattr(model, "get_booster"):
        feature_order = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        feature_order = list(model.feature_names_in_)
    else:
        feature_order = list(input_df.columns)

    for col in feature_order:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[feature_order]

    # Cast to numeric types friendly for XGBoost
    for col in input_df.select_dtypes(include=["float64"]).columns:
        input_df[col] = input_df[col].astype(np.float32)
    for col in input_df.select_dtypes(include=["int64"]).columns:
        input_df[col] = input_df[col].astype(np.int32)

    if st.button("Predict Claim"):
        proba = model.predict_proba(input_df)[0, 1]
        prediction = int(proba >= 0.5)
        st.write(f"**Predicted probability of claim:** {proba:.4f}")
        st.write(f"**Predicted class (0 = No Claim, 1 = Claim):** {prediction}")
