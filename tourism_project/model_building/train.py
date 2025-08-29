# ==============================
# Tourism Package Purchase Prediction - Model Training
# ==============================

# Data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Model training & evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Model serialization
import joblib

# Hugging Face Hub
import os
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow tracking
import mlflow

# ------------------------------
# MLflow Setup
# ------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# ------------------------------
# Dataset paths (Hugging Face Datasets)
# ------------------------------
Xtrain_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/Xtest.csv"
ytrain_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/ytrain.csv"
ytest_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/ytest.csv"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze().astype(int)
ytest = pd.read_csv(ytest_path).squeeze().astype(int)

print("Data loaded successfully.")

# ------------------------------
# Preprocessing setup
# ------------------------------

# Separate numeric & categorical features
numeric_features = [
    'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar',
    'NumberOfTrips', 'NumberOfChildrenVisiting',
    'MonthlyIncome', 'PitchSatisfactionScore',
    'NumberOfFollowups', 'DurationOfPitch'
]

categorical_features = [
    'TypeofContact', 'CityTier', 'Occupation', 'Gender',
    'MaritalStatus', 'Passport', 'OwnCar', 'Designation',
    'ProductPitched'
]

# Compute class weights (handle imbalance)
class_counts = ytrain.value_counts()
class_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
print("Class counts:", class_counts.to_dict())
print("Class weight:", class_weight)

# Define preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# ------------------------------
# Model Setup
# ------------------------------
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.5, 0.7, 1.0]
}

# Build pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------------------
# Training with MLflow tracking
# ------------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    print("Training complete.")

    # Best model
    best_model = grid_search.best_estimator_

    # ------------------------------
    # Custom classification threshold
    # ------------------------------
    classification_threshold = 0.45
    mlflow.log_param("classification_threshold", classification_threshold)

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]

    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # ------------------------------
    # Log metrics
    # ------------------------------
    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"]
    })

    # ------------------------------
    # Save and Upload Model
    # ------------------------------
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Best model saved at: {model_path}")

    # Upload model to Hugging Face
    repo_id = "SudeendraMG/tourism_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Hugging Face repo '{repo_id}' exists.")
    except RepositoryNotFoundError:
        print(f"Creating new Hugging Face repo: {repo_id}")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print(f"Model uploaded to Hugging Face Hub at: {repo_id}")
