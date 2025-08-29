%%writefile tourism_project/model_building/train.py
# ------------------------------
# Imports
# ------------------------------
import os
import joblib
import mlflow
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# ------------------------------
# MLflow Setup
# ------------------------------
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-training-experiment")

api = HfApi()

# ------------------------------
# Dataset Paths
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

# ------------------------------
# Identify numeric & categorical features
# ------------------------------
numeric_features = Xtrain.select_dtypes(include="number").columns.tolist()
categorical_features = Xtrain.select_dtypes(exclude="number").columns.tolist()

# ------------------------------
# Handle class imbalance
# ------------------------------
class_counts = ytrain.value_counts()
class_weight = class_counts.get(0, 1) / class_counts.get(1, 1)
print("Scale_pos_weight:", class_weight)

# ------------------------------
# Preprocessing Pipeline
# ------------------------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    use_label_encoder=False,
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

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------------------
# Start MLflow Run
# ------------------------------
with mlflow.start_run():
    # Grid Search
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="roc_auc")
    grid_search.fit(Xtrain, ytrain)

    # Log CV results
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_f1", mean_score)
            mlflow.log_metric("std_test_f1", std_score)

    # Best model
    best_model = grid_search.best_estimator_
    mlflow.log_params(grid_search.best_params_)

    # --------------------------
    # Custom threshold
    # --------------------------
    classification_threshold = 0.45

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # --------------------------
    # Reports & Metrics
    # --------------------------
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # ROC-AUC
    train_auc = roc_auc_score(ytrain, y_pred_train_proba)
    test_auc = roc_auc_score(ytest, y_pred_test_proba)

    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "train_auc": train_auc,
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score'],
        "test_auc": test_auc
    })

    print("Training complete.")
    print("Train AUC:", train_auc, " Test AUC:", test_auc)

    # --------------------------
    # Plot ROC curve
    # --------------------------
    fpr, tpr, _ = roc_curve(ytest, y_pred_test_proba)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Test Data")
    plt.legend()
    plt.savefig("roc_curve.png")

    # Log ROC curve
    mlflow.log_artifact("roc_curve.png")

    # --------------------------
    # Save model
    # --------------------------
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # --------------------------
    # Upload to Hugging Face Hub
    # --------------------------
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
    print("Upload complete.")
