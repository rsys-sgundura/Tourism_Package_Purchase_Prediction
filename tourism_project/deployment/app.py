%%writefile tourism_project/deployment/app.py
import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib
import numpy as np

# ------------------------------------------------------
# Load trained model
# ------------------------------------------------------
MODEL_REPO = "SudeendraMG/tourism_model"
MODEL_FILE = "best_tourism_model_v1.joblib"
ENCODER_FILE = "label_encoders.pkl"   # <- Added encoders

THRESHOLD = 0.45  # align with training/evaluation threshold

# Download and load model
model_path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
model = joblib.load(model_path)

# Download and load saved label encoders
encoder_path = hf_hub_download(repo_id=MODEL_REPO, filename=ENCODER_FILE)
encoders = joblib.load(encoder_path)

# ------------------------------------------------------
# UI
# ------------------------------------------------------
st.title("Tourism Package Purchase Prediction")
st.write("""
Predict whether a customer is likely to purchase a tourism package based on
their profile and interaction details.
""")

# --- Customer Details ---
age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
typeof_contact = st.selectbox("TypeofContact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("CityTier", [1, 2, 3], index=0)
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer", "Small Business",
                                         "Large Business", "Student", "Retired", "Other"], index=0)
gender = st.selectbox("Gender", ["Male", "Female"], index=0)
number_of_person_visiting = st.number_input("NumberOfPersonVisiting", min_value=1, max_value=20, value=2, step=1)
preferred_property_star = st.selectbox("PreferredPropertyStar", [1, 2, 3, 4, 5], index=2)
marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"], index=0)
number_of_trips = st.number_input("NumberOfTrips (avg per year)", min_value=0, max_value=50, value=2, step=1)
passport = st.selectbox("Passport", ["No", "Yes"], index=0)
own_car = st.selectbox("OwnCar", ["No", "Yes"], index=0)
number_of_children_visiting = st.number_input("NumberOfChildrenVisiting (below age 5)", min_value=0, max_value=10, value=0, step=1)
designation = st.selectbox("Designation", ["Executive", "Senior Executive", "Manager",
                                           "Senior Manager", "AVP", "VP", "Other"], index=0)
monthly_income = st.number_input("MonthlyIncome (INR)", min_value=1000, max_value=2_000_000, value=50_000, step=1000)

# --- Customer Interaction Data ---
st.header("Customer Interaction Data")
pitch_satisfaction_score = st.selectbox("PitchSatisfactionScore", [1, 2, 3, 4, 5], index=3)
product_pitched = st.selectbox("ProductPitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"], index=2)
number_of_followups = st.number_input("NumberOfFollowups", min_value=0, max_value=30, value=2, step=1)
duration_of_pitch = st.number_input("DurationOfPitch (minutes)", min_value=0, max_value=240, value=15, step=1)

# ------------------------------------------------------
# Assemble input (exclude CustomerID, match training features)
# ------------------------------------------------------
input_data = pd.DataFrame([{
    "Age": int(age),
    "TypeofContact": typeof_contact,
    "CityTier": int(city_tier),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(number_of_person_visiting),
    "PreferredPropertyStar": int(preferred_property_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": int(number_of_trips),
    "Passport": passport,   # keep as string -> will encode below
    "OwnCar": own_car,      # keep as string -> will encode below
    "NumberOfChildrenVisiting": int(number_of_children_visiting),
    "Designation": designation,
    "MonthlyIncome": float(monthly_income),
    "PitchSatisfactionScore": int(pitch_satisfaction_score),
    "ProductPitched": product_pitched,
    "NumberOfFollowups": int(number_of_followups),
    "DurationOfPitch": float(duration_of_pitch),
}])

st.markdown("**Preview of input before encoding**")
st.dataframe(input_data)

# ------------------------------------------------------
# Apply saved LabelEncoders to categorical columns
# ------------------------------------------------------
for col, le in encoders.items():
    if col in input_data.columns:
        try:
            input_data[col] = le.transform(input_data[col])
        except ValueError as e:
            st.error(f"Value '{input_data[col].iloc[0]}' not seen during training for column '{col}'")
            st.stop()

st.markdown("**Input after encoding (final features to model)**")
st.dataframe(input_data)

# ------------------------------------------------------
# Predict
# ------------------------------------------------------
if st.button("Predict Purchase"):
    proba = model.predict_proba(input_data)[:, 1][0]
    pred = int(proba >= THRESHOLD)

    label = "Will Purchase (1)" if pred == 1 else "Will Not Purchase (0)"
    st.subheader("Prediction")
    st.write(f"**Class:** {label}")
    st.write(f"**Probability of Purchase (class=1):** {proba:.3f}")
    st.write(f"**Decision Threshold used:** {THRESHOLD:.2f}")
