# ------------------------------------------------------
# Data Preparation Script for Tourism Dataset
# ------------------------------------------------------
# This script:
#  1. Loads dataset from Hugging Face Hub
#  2. Cleans and preprocesses the data
#  3. Handles missing values and outliers
#  4. Encodes categorical variables
#  5. Normalizes numerical features
#  6. Splits the dataset into train and test sets
#  7. Saves the processed files locally
#  8. Uploads processed files back to Hugging Face Hub
# ------------------------------------------------------

# Import required libraries
import pandas as pd
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from huggingface_hub import login, HfApi

# Initialize Hugging Face API using token stored in environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load dataset from Hugging Face Hub
DATASET_PATH = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ------------------------------------------------------
# Step 1: Drop unique identifier columns
# ------------------------------------------------------
# Unique ID/index columns do not provide predictive value
unique_cols = [col for col in df.columns if df[col].is_unique]
if unique_cols:
    df.drop(columns=unique_cols, inplace=True)
    print(f"Dropped unique columns: {unique_cols}")

# ------------------------------------------------------
# Step 2: Separate numeric and categorical columns
# ------------------------------------------------------
num_cols = df.select_dtypes(include="number").columns.tolist()
cat_cols = df.select_dtypes(exclude="number").columns.tolist()

# ------------------------------------------------------
# Step 3: Handle missing values
# ------------------------------------------------------
# - For numeric: replace missing with median
# - For categorical: replace missing with most frequent value
if num_cols:
    num_imputer = SimpleImputer(strategy="median")
    df[num_cols] = num_imputer.fit_transform(df[num_cols])
if cat_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("Missing values imputed.")

# ------------------------------------------------------
# Step 4: Outlier treatment (IQR method)
# ------------------------------------------------------
# Any value outside 1.5 * IQR range will be capped at boundary
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

print("Outliers treated with IQR capping.")

# ------------------------------------------------------
# Step 5: Encode categorical variables
# ------------------------------------------------------
# LabelEncoder assigns integer values to string categories
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

print("Categorical variables encoded using LabelEncoder.")

# ------------------------------------------------------
# Step 6: Normalize numerical columns
# ------------------------------------------------------
# StandardScaler: mean = 0, variance = 1
if num_cols:
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("Numerical columns normalized.")

# ------------------------------------------------------
# Step 7: Separate target variable
# ------------------------------------------------------
# Replace "Target" with the actual target column of tourism dataset
target_col = "ProdTaken"  # <-- Update this with correct target column name
if target_col in df.columns:
    X = df.drop(columns=[target_col])  # Features
    y = df[target_col]                 # Target
else:
    raise ValueError("Target column not found. Please update 'target_col'.")

# ------------------------------------------------------
# Step 8: Train-test split
# ------------------------------------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------
# Step 9: Save processed data locally
# ------------------------------------------------------
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print("Train-test data prepared and saved locally as CSVs.")

# ------------------------------------------------------
# Step 10: Upload processed files to Hugging Face Hub
# ------------------------------------------------------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="SudeendraMG/tourism-package-purchase-prediction",
        repo_type="dataset",
    )

print("Processed data uploaded successfully to Hugging Face Hub.")
