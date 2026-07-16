"""
Project 2: Fraud Detection Pipeline (Supervised Learning)
DecodeLabs Internship
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE


# Step 1: Load Dataset

def load_data(path="creditcard.csv"):
    print(f"[1/7] Loading dataset from '{path}' ...")
    df = pd.read_csv(path)
    print(f"      Shape: {df.shape}")
    print(f"      Fraud percentage: {round(df['Class'].mean() * 100, 3)}%")
    return df


# Step 2: Split Features and Target

def split_features_target(df):
    print("[2/7] Splitting features (X) and target (y) ...")
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


# Step 3: Train/Test Split (stratified, BEFORE any resampling/scaling)

def split_train_test(X, y):
    print("[3/7] Performing stratified train/test split ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Train size: {X_train.shape} | Test size: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# Step 4: Build Pipelines

def build_pipelines():
    print("[4/7] Building Logistic Regression and Random Forest pipelines ...")

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("smote", SMOTE(random_state=42)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    rf_pipeline = Pipeline(
        [
            ("smote", SMOTE(random_state=42)),
            ("classifier", RandomForestClassifier(random_state=42, n_jobs=-1)),
        ]
    )

    return lr_pipeline, rf_pipeline


# Step 5: Train Models

def train_models(lr_pipeline, rf_pipeline, X_train, y_train):
    print("[5/7] Training Logistic Regression pipeline ...")
    lr_pipeline.fit(X_train, y_train)

    print("      Training Random Forest pipeline ...")
    rf_pipeline.fit(X_train, y_train)

    return lr_pipeline, rf_pipeline


# Step 6: Hyperparameter Tuning (lightweight GridSearchCV on Random Forest)

def tune_random_forest(rf_pipeline, X_train, y_train):
    print("[6/7] Running GridSearchCV on Random Forest (this may take a few minutes) ...")

    param_grid = {
        "classifier__n_estimators": [100],
        "classifier__max_depth": [10, None],
        "smote__k_neighbors": [5],
    }

    grid_search = GridSearchCV(
        rf_pipeline,
        param_grid,
        scoring="roc_auc",
        cv=2,
        n_jobs=-1,
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    print(f"      Best Parameters: {grid_search.best_params_}")
    print(f"      Best ROC-AUC (CV): {round(grid_search.best_score_, 4)}")

    return grid_search.best_estimator_


# Step 7: Evaluate Models and Save Results

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f"      {name} -> Precision: {round(precision, 4)} | "
          f"Recall: {round(recall, 4)} | ROC-AUC: {round(roc_auc, 4)}")

    return {
        "Model": name,
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "ROC_AUC": round(roc_auc, 4),
        "True_Positives": tp,
        "False_Positives": fp,
        "True_Negatives": tn,
        "False_Negatives": fn,
    }


def main():
    # 1. Load data
    df = load_data("creditcard.csv")

    # 2. Feature/target split
    X, y = split_features_target(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 4. Build pipelines
    lr_pipeline, rf_pipeline = build_pipelines()

    # 5. Train models
    lr_pipeline, rf_pipeline = train_models(lr_pipeline, rf_pipeline, X_train, y_train)

    # 6. Tune Random Forest
    best_rf_model = tune_random_forest(rf_pipeline, X_train, y_train)

    # 7. Evaluate all models
    print("[7/7] Evaluating all models on the untouched test set ...")
    results = []
    results.append(evaluate_model("Logistic Regression", lr_pipeline, X_test, y_test))
    results.append(evaluate_model("Random Forest", rf_pipeline, X_test, y_test))
    results.append(evaluate_model("Best Tuned Random Forest", best_rf_model, X_test, y_test))

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_path = "fraud_detection_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nDone! Results saved to '{output_path}'")
    print(results_df)


if __name__ == "__main__":
    main()