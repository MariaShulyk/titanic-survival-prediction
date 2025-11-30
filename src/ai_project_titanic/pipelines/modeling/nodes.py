import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from pycaret.classification import setup, compare_models, pull, predict_model


TARGET_COL = "Survived"


def _split_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def _compute_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_proba: np.ndarray | None = None
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def train_baseline(
    train_data: pd.DataFrame, val_data: pd.DataFrame
) -> tuple[DummyClassifier, Dict[str, Any]]:
    X_train, y_train = _split_X_y(train_data)
    X_val, y_val = _split_X_y(val_data)

    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    metrics = _compute_metrics(y_val, y_pred, y_proba=None)
    metrics["model_type"] = "baseline_dummy_most_frequent"

    return model, metrics


def train_automl(

    train_data: pd.DataFrame, val_data: pd.DataFrame
) -> tuple[object, pd.DataFrame, Dict[str, Any]]:

    setup(
        data=train_data,
        target=TARGET_COL,
        session_id=42,
        verbose=False,
        html=False,
    )

    best_model = compare_models(n_select=1, sort="Accuracy")

    if isinstance(best_model, list):
        if len(best_model) == 0:
            raise RuntimeError("compare_models zwrócił pustą listę modeli – coś poszło nie tak z treningiem.")
        best_model = best_model[0]

    results_df = pull().reset_index(drop=True)

    preds = predict_model(best_model, data=val_data.copy())
    y_true = val_data[TARGET_COL]
    y_pred = preds["prediction_label"]

    proba_col = None
    for col in preds.columns:
        if "score" in col or "prob" in col:
            proba_col = col
            break

    if proba_col is not None:
        y_proba = preds[proba_col].values
    else:
        y_proba = None

    metrics = _compute_metrics(y_true, y_pred, y_proba=y_proba)
    metrics["model_type"] = "pycaret_automl_best"

    return best_model, results_df, metrics


def train_custom(
    train_data: pd.DataFrame, val_data: pd.DataFrame
) -> tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Własny model ML: RandomForestClassifier z prostym preprocessingiem:
    - cechy numeryczne,
    - one-hot encoding dla wybranych zmiennych kategorycznych (Sex, Embarked).
    Trenujemy na train, oceniamy na val.
    """
    X_train, y_train = _split_X_y(train_data)
    X_val, y_val = _split_X_y(val_data)

    categorical_cols = [col for col in ["Sex", "Embarked"] if col in X_train.columns]
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()

    X_train_num = X_train[numeric_cols].copy()
    X_val_num = X_val[numeric_cols].copy()

    if categorical_cols:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_val_cat = pd.get_dummies(X_val[categorical_cols], drop_first=True)

        X_val_cat = X_val_cat.reindex(columns=X_train_cat.columns, fill_value=0)
    else:
        X_train_cat = pd.DataFrame(index=X_train.index)
        X_val_cat = pd.DataFrame(index=X_val.index)

    X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    X_val_final = pd.concat([X_val_num, X_val_cat], axis=1)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_val_final)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_val_final)[:, 1]
    else:
        y_proba = None

    metrics = _compute_metrics(y_val, y_pred, y_proba=y_proba)
    metrics["model_type"] = "random_forest_custom_with_dummies"

    return model, metrics



def evaluate_models(
    baseline_metrics: Dict[str, Any],
    automl_metrics: Dict[str, Any],
    custom_metrics: Dict[str, Any],
    baseline_model: DummyClassifier,
    automl_model: object,
    custom_model: RandomForestClassifier,
) -> tuple[Dict[str, Any], object]:

    all_metrics = {
        "baseline": baseline_metrics,
        "automl": automl_metrics,
        "custom": custom_metrics,
    }

    best_name = None
    best_score = -1.0

    for name, m in all_metrics.items():
        roc_auc = m.get("roc_auc", float("nan"))
        if not np.isnan(roc_auc) and roc_auc > best_score:
            best_score = roc_auc
            best_name = name

    if best_name == "baseline":
        best_model = baseline_model
    elif best_name == "automl":
        best_model = automl_model
    else:
        best_model = custom_model

    comparison = {
        "baseline": baseline_metrics,
        "automl": automl_metrics,
        "custom": custom_metrics,
        "best_model_name": best_name,
        "best_model_roc_auc": best_score,
    }

    return comparison, best_model
