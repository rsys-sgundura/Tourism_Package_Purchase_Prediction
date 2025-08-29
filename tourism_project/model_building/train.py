# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report
# for model serialization
import joblib
# for hugging face space authentication to upload files
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
ytrain = pd.read_csv(ytrain_path).squeeze().astype(int)
ytest = pd.read_csv(ytest_path).squeeze().astype(int)

# Ensure stratified split in case uploaded CSVs are unbalanced
X = pd.concat([Xtrain, Xtest], axis=0)
y = pd.concat([ytrain, ytest], axis=0)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight, random_state=42, use_label_encoder=False, eval_metric="logloss"
)

# Hyperparameter grid for tuning
param_grid = {
    'xgbclassifier__n_estimators': [50, 100, 150],
    'xgbclassifier__max_depth': [3, 5, 7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],
    'xgbclassifier__subsample': [0.7, 0.9, 1.0],
}

# Model pipeline
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ------------------------------------------------------
# Helper: Safe access for missing class labels
# ------------------------------------------------------
def safe_get(report, label, metric):
    return report[label][metric] if label in report else 0.0

# ------------------------------------------------------
# Start MLflow run
# ------------------------------------------------------
with mlflow.start_run():
    # Grid search with cross-validation
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    # Log each parameter combination with mean/std F1 scores
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        param_set = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]

        with mlflow.start_run(nested=True):
            mlflow.log_params(param_set)
            mlflow.log_metric("mean_test_f1", mean_score)
            mlflow.log_metric("std_test_f1", std_score)

    # Log best parameters from grid search
    mlflow.log_params(grid_search.best_params_)

    # Retrieve best model
    best_model = grid_search.best_estimator_

    # Apply custom classification threshold
    classification_threshold = 0.45
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Reports
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics safely
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": safe_get(train_report, "1", "precision"),
        "train_recall": safe_get(train_report, "1", "recall"),
        "train_f1-score": safe_get(train_report, "1", "f1-score"),
        "test_accuracy": test_report['accuracy'],
        "test_precision": safe_get(test_report, "1", "precision"),
        "test_recall": safe_get(test_report, "1", "recall"),
        "test_f1-score": safe_get(test_report, "1", "f1-score")
    })

    # Save the model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log model as MLflow artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # ------------------------------------------------------
    # Upload best model to Hugging Face Hub
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
