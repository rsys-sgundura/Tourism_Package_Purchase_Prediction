# ------------------------------------------------------
# Tourism Package Purchase Prediction - Model Training
# ------------------------------------------------------

# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
# for model serialization
import joblib
# for hugging face hub uploads
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

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

# Load datasets
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze("columns").astype(int)
ytest = pd.read_csv(ytest_path).squeeze("columns").astype(int)

print("Data loaded successfully.")
print("Training set class distribution:\n", ytrain.value_counts())
print("Test set class distribution:\n", ytest.value_counts())

# ------------------------------------------------------
# Identify numeric vs categorical features
# ------------------------------------------------------
# At this stage all categorical columns were label-encoded in prep.py,
# so they are integers but semantically categorical.
# -> We should NOT scale these categorical-int columns.
categorical_int_cols = [
    "Gender",
    "NumberOfFollowups",
    "PreferredPropertyStar",
    "Passport",
    "CityTier",
    "NumberOfPersonVisiting",
    "PitchSatisfactionScore",
    "OwnCar",
    "NumberOfChildrenVisiting",
    "Designation",  # also label-encoded
    "MaritalStatus" # also label-encoded
]

numeric_features = [col for col in Xtrain.select_dtypes(include="number").columns 
                    if col not in categorical_int_cols]

print("Numeric features:", numeric_features)
print("Categorical (encoded int) features:", categorical_int_cols)

# ------------------------------------------------------
# Handle class imbalance using scale_pos_weight
# ------------------------------------------------------
class_counts = ytrain.value_counts()
class_weight = class_counts[0] / class_counts[1]
print("Scale_pos_weight:", class_weight)

# ------------------------------------------------------
# Define preprocessing pipeline
# ------------------------------------------------------
# Only scale numeric continuous features
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    remainder="passthrough"   # keep categorical ints as-is
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric="logloss"  # avoids "use_label_encoder" warning
)

# Hyperparameter grid for tuning
param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.8, 1.0],
    'xgbclassifier__subsample': [0.9, 1.0],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------------------------------------------
# Start MLflow run
# ------------------------------------------------------
with mlflow.start_run():
    # Grid search with cross-validation
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Log best parameters
    mlflow.log_params(grid_search.best_params_)

    # Best model
    best_model = grid_search.best_estimator_

    # Apply custom classification threshold
    classification_threshold = 0.45
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
    test_report = classification_report(ytest, y_pred_test, output_dict=True, zero_division=0)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report.get('1', {}).get('precision', 0),
        "train_recall": train_report.get('1', {}).get('recall', 0),
        "train_f1-score": train_report.get('1', {}).get('f1-score', 0),
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report.get('1', {}).get('precision', 0),
        "test_recall": test_report.get('1', {}).get('recall', 0),
        "test_f1-score": test_report.get('1', {}).get('f1-score', 0)
    })

    # ROC AUC
    if len(set(ytest)) > 1:
        auc = roc_auc_score(ytest, y_pred_test_proba)
        mlflow.log_metric("test_roc_auc", auc)
        print("Test ROC AUC:", auc)

    # Save the model
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log model in MLflow
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

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

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
