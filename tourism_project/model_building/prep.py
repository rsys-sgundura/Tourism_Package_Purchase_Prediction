# ------------------------------------------------------
# Tourism Package Purchase Prediction - Data Preparation
# ------------------------------------------------------


# for data manipulation
import pandas as pd
# for train-test split
from sklearn.model_selection import train_test_split
# for encoding
from sklearn.preprocessing import LabelEncoder
# for saving encoders
import joblib
# for hugging face hub
import os
from huggingface_hub import HfApi

# ------------------------------------------------------
# Constants
# ------------------------------------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/tourism.csv"
REPO_ID = "SudeendraMG/tourism-package-purchase-prediction"

# ------------------------------------------------------
# Load dataset
# ------------------------------------------------------
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ------------------------------------------------------
# Basic cleaning
# ------------------------------------------------------
# Drop CustomerID (unique identifier not used for training)
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Fix inconsistent gender values
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace("Fe Male", "Female")

# ------------------------------------------------------
# Convert columns to category
# ------------------------------------------------------
# All object columns → category
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category")

# Explicit numeric-to-category conversions
cat_cols = [
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "Passport",
    "CityTier",
    "NumberOfPersonVisiting",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
]
for col in cat_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

# ------------------------------------------------------
# Label Encoding
# ------------------------------------------------------
encoders = {}
for col in df.select_dtypes(include="category").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("Categorical columns label encoded.")

# ------------------------------------------------------
# Split into train/test
# ------------------------------------------------------
target_col = "ProdTaken"  # Target variable in tourism dataset
X = df.drop(columns=[target_col])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save splits
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Save encoders
joblib.dump(encoders, "label_encoders.pkl")
print("Train/test splits and label encoders saved locally.")

# ------------------------------------------------------
# Upload files to Hugging Face Hub
# ------------------------------------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv", "label_encoders.pkl"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=REPO_ID,
        repo_type="dataset",
    )

print("All files uploaded to Hugging Face dataset repo successfully.")
