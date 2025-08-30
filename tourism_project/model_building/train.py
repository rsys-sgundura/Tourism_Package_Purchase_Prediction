# ------------------------------------------------------
# Tourism Package Purchase Prediction - Model Training
# ------------------------------------------------------

# for data manipulation
import pandas as pd
import joblib
import xgboost as xgb
import mlflow
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

# for hugging face space authentication to upload files
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ------------------------------------------------------
# Setup MLflow experiment
# ------------------------------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# ------------------------------------------------------
# Dataset paths (from Hugging Face dataset repo)
# ------------------------------------------------------
Xtrain_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/Xtrain.csv"
Xtest_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/Xtest.csv"
ytrain_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/ytrain.csv"
ytest_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/ytest.csv"
encoders_path = "hf://datasets/SudeendraMG/tourism-package-purchase-prediction/label_encoders.pkl"

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze("columns").astype(int)
ytest = pd.read_csv(ytest_path).squeeze("columns").astype(int)

# Load label encoders (created in prep.py and uploaded)
import joblib
encoders = joblib.load(encoders_path)

print("Data + encoders loaded successfully.")
print("Training set class distribution:\n", ytrain.value_counts())
print("Test set class distribution:\n", ytest.value_counts())

# ------------------------------------------------------
# Identify numeric and categorical features
# ------------------------------------------------------
numeric_features = Xtrain.select_dtypes(include="number").columns.tolist()
categorical_features = Xtrain.select_dtypes(exclude="number").columns.tolist()

# ------------------------------------------------------
# Handle class imbalance using scale_pos_weight
# ------------------------------------------------------
class_counts = ytrain.value_counts()
class_weight = class_counts[0] / class_counts[1]
print("Scale_pos_weight:", class_weight)

# ------------------------------------------------------
# Define preprocessing pipeline
# ------------------------------------------------------
# (Since categorical vars were already label-encoded, only scale numerics)
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="passthrough"
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"
)

# Hyperparameter grid
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
    'xgbclassifier__subsample': [0.9, 1.0],
}

# Pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------------------------------------------
# Start MLflow run
# ------------------------------------------------------
with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    # Apply custom threshold
    classification_threshold = 0.45
    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Classification report
    test_report = classification_report(ytest, y_pred_test, output_dict=True, zero_division=0)

    # Log metrics
    mlflow.log_metrics({
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report.get('1', {}).get('precision', 0),
        "test_recall": test_report.get('1', {}).get('recall', 0),
        "test_f1-score": test_report.get('1', {}).get('f1-score', 0),
    })

    # Optional ROC AUC
    if len(set(ytest)) > 1:
        auc = roc_auc_score(ytest, y_pred_test_proba)
        mlflow.log_metric("test_roc_auc", auc)
        print("Test ROC AUC:", auc)

    # ------------------------------------------------------
    # Save model + encoders
    # ------------------------------------------------------
    model_path = "best_tourism_model_v1.joblib"
    encoder_path = "label_encoders.pkl"

    joblib.dump(best_model, model_path)
    joblib.dump(encoders, encoder_path)

    mlflow.log_artifact(model_path, artifact_path="model")
    mlflow.log_artifact(encoder_path, artifact_path="encoders")

    print(f"Model saved as {model_path}, encoders saved as {encoder_path}")

    # ------------------------------------------------------
    # Upload to Hugging Face Hub
    # ------------------------------------------------------
    repo_id = "SudeendraMG/tourism_model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    # Upload both files
    for file_path in [model_path, encoder_path]:
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file_path,
            repo_id=repo_id,
            repo_type=repo_type,
        )
