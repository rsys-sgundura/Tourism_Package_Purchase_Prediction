import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the model from the Model Hub
model_path = hf_hub_download(repo_id="MainiSandeep1987/tourism-prediction-model", filename="best_tourism_prediction_model_v1.joblib")


# Load the model
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Tourism Package Prediction App")
st.write("The Tourism Package Prediction App is an internal tool for \"Visit with Us\" i.e. a leading travel company management & sales that predicts whether a customer will purchase the newly introduced Wellness Tourism Package before contacting them based on their details.")
st.write("Kindly enter the customer details to check whether they are likely to Opt In for tourism package.")


# Collect user input
Age = st.number_input("Age (customer's age in years)", min_value=18.0, max_value=110.0, value=18.0,step=1.0)
CityTier = st.selectbox("The city category based on development, population, and living standards (Tier 1 > Tier 2 > Tier 3)",
                        ["Tier 1", "Tier 2", "Tier 3"])
NumberOfPersonVisiting = st.number_input("Total number of people accompanying the customer on the trip", min_value=0, max_value=30, value=0,step=1)
PreferredPropertyStar = st.number_input("Preferred hotel rating by the customer",min_value=1.0, max_value=7.0, value=3.0,step=1.0)
NumberOfTrips = st.number_input("Average number of trips the customer takes annually",min_value=0.0, value=1.0,step=1.0)
Passport = st.selectbox("Whether the customer holds a valid passport ?",["Yes", "No"])
OwnCar = st.selectbox("Whether the customer owns a car ?",["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of children below age 5 accompanying the customer",min_value=0.0, value=0.0,step=1.0)
MonthlyIncome = st.number_input("Gross monthly income of the customer", min_value=0.0, value=5000.0)
PitchSatisfactionScore = st.number_input("Score indicating the customer's satisfaction with the sales pitch", min_value=1, value=1,max_value=5,step=1)
NumberOfFollowups = st.number_input("Total number of follow-ups by the salesperson after the sales pitch.",min_value=0.0, value=1.0,step=1.0)
DurationOfPitch = st.number_input("Duration of the sales pitch delivered to the customer.",min_value=1.0, value=1.0,step=1.0)

TypeofContact = st.selectbox("The method by which the customer was contacted",["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Customer's occupation",["Salaried", "Small Business","Large Business","Free Lancer"])
Gender = st.selectbox("Gender of the customer",["Male", "Female"])
MaritalStatus = st.selectbox("Marital status of the customer",["Married", "Divorced","Unmarried","Single"])
Designation = st.selectbox("Customer's designation in their current organization",["Executive", "Manager","Senior Manager", "AVP","VP"])
ProductPitched = st.selectbox("The type of product pitched to the customer",["Basic", "Deluxe","Standard","Super Deluxe","King"])

citytier_mapping = {'Tier 1':1,'Tier 2':2,'Tier 3':3}
                                                                             

                                                                             
# Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Age': Age,
    'CityTier': citytier_mapping[CityTier],
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'PreferredPropertyStar': PreferredPropertyStar,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfFollowups': NumberOfFollowups,
    'DurationOfPitch': DurationOfPitch,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation,
    'ProductPitched': ProductPitched
}])

# Set the classification threshold
classification_threshold = 0.45

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Opted For Tourism Package" if prediction == 1 else "Not Opted For Tourism Package"
    st.write(f"Based on the information provided, the customer is likely to {result}.")
