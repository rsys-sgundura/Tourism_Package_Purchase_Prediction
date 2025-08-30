# ------------------------------------------------------
# Data preparation script for the Tourism purchase model
#
# Why this script exists:
# - Enforces consistent cleaning and encoding so training and serving
#   see the same feature space.
# - Produces train/test CSVs and uploads them to the HF dataset repo
#   for reproducibility and downstream pipelines.
# ------------------------------------------------------

# for data manipulation
import pandas as pd
import sklearn  # not directly used, but keeps sklearn version pinned in envs that lint imports
# for creating a folder / env access
import os
# for data preprocessing and splitting
from sklearn.model_selection import train_test_split
# for converting categorical text to numeric codes (model-ready)
from sklearn.preprocessing import LabelEncoder
# for Hugging Face Hub uploads
from huggingface_hub import HfApi

# ------------------------------------------------------
# Config: HF auth & dataset location
# ------------------------------------------------------
# We use a token from the environment so secrets never live in code.
api = HfApi(token=os.getenv("HF_TOKEN"))

# Source dataset lives in an HF dataset repo; using the hf:// protocol
# avoids manual download and ensures we read the exact committed artifact.
DATASET_PATH = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/tourism.csv"

# ------------------------------------------------------
# Step 1: Load dataset
# ------------------------------------------------------
# Keep raw load simple and explicit; any I/O failure will raise early.
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# ------------------------------------------------------
# Step 2: Clean inconsistent values & set categorical dtypes
# ------------------------------------------------------
# 2a) Fix inconsistent 'Gender' label:
#     The dataset sometimes has "Fe Male"; we normalize it to "Female"
#     so downstream encoders see a single, consistent category.
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# 2b) Convert all object columns to pandas 'category' dtype:
#     - More memory efficient than plain object strings
#     - Makes downstream intent explicit (treated as categorical, not free text)
obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
for col in obj_cols:
    df[col] = df[col].astype("category")

# 2c) Explicitly treat selected integer-coded columns as categorical:
#     These represent discrete options, not true continuous numeric measures.
cat_int_cols = [
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "Passport",
    "CityTier",
    "NumberOfPersonVisiting",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
]
for col in cat_int_cols:
    if col in df.columns:
        df[col] = df[col].astype("category")

print("Categorical conversion completed.")

# ------------------------------------------------------
# Step 3: Label Encode all categorical columns
# ------------------------------------------------------
# Why LabelEncoder here?
# - We’re preparing plain CSVs (no sklearn pipeline persisted with OneHotEncoder),
#   so we need numeric values in the saved files.
# - LabelEncoder gives deterministic, compact integer codes per column.
# Note: The actual numeric codes are arbitrary (alphabetical order); that’s fine
# as long as the same encoders are used consistently between train and serve.
cat_cols = df.select_dtypes(include=["category"]).columns.tolist()
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
print("Label Encoding applied to categorical columns.")

# ------------------------------------------------------
# Step 4: Separate features/target & drop identifiers
# ------------------------------------------------------
# Target column per business problem statement.
target_col = "ProdTaken"

# Drop purely-identifying columns from features; they add leakage/noise.
drop_cols = ["CustomerID"]

# Build X and y; errors='ignore' makes code robust if a column is missing.
X = df.drop(columns=[target_col] + drop_cols, errors="ignore")
y = df[target_col]

# ------------------------------------------------------
# Step 5: Train-test split with stratification
# ------------------------------------------------------
# Stratify preserves the target class distribution in both splits,
# critical for imbalanced classification.
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# (Optional) quick sanity check prints for class balance
print("Train target distribution:\n", ytrain.value_counts(dropna=False).to_string())
print("Test target distribution:\n", ytest.value_counts(dropna=False).to_string())

# ------------------------------------------------------
# Step 6: Persist splits as flat files
# ------------------------------------------------------
# CSVs are simple interop artifacts that any training/serving job can consume.
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train/test CSVs written locally.")

# ------------------------------------------------------
# Step 7: Upload artifacts to HF dataset repo
# ------------------------------------------------------
# Keeping the splits in the same dataset repo ensures reproducibility and
# easy consumption across CI, notebooks, and Spaces.
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # just the filename
        repo_id="SudeendraMG/tourism-package-purchase-prediction",
        repo_type="dataset",
    )
print("Train-test CSVs uploaded to Hugging Face dataset repo.")
