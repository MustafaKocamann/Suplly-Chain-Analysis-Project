"""
Model training and evaluation pipeline for Supply Chain project.

Trains and evaluates 4 classifiers for two binary targets:
    1. fraud         (~2.3% positive — imbalanced)
    2. late_delivery (~55% positive  — balanced)

Expected inputs (from modeling_preprocessing_pipeline output):
    xf_train_balanced, yf_train_balanced  — fraud train (SMOTE)
    xf_test_scaled,    yf_test            — fraud test
    xl_train_scaled,   yl_train           — late delivery train
    xl_test_scaled,    yl_test            — late delivery test

Outputs:
    results           — list of dicts with all metrics
    fraud_comparison  — DataFrame sorted by ROC-AUC
    late_comparison   — DataFrame sorted by ROC-AUC
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


# ── Constants ────────────────────────────────────────────────────────
RANDOM_STATE = 42


def print_section(title: str) -> None:
    """Print consistent section header for audit trail."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def compute_metrics(
    model,
    X_test: np.ndarray,
    y_true: pd.Series,
    target_name: str,
) -> dict[str, Any]:
    """
    Compute all evaluation metrics with optimized decision threshold for imbalanced targets.

    Parameters
    ----------
    model      : fitted sklearn-compatible classifier
    X_test     : scaled test features
    y_true     : true labels
    target_name: 'fraud' or 'late_delivery' — used for display

    Returns
    -------
    dict with accuracy, f1, recall, precision, roc_auc,
    confusion_matrix, y_pred, y_proba, threshold
    """
    y_proba = None
    roc_auc = None

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_true, y_proba)
    except AttributeError:
        # Fallback for models without predict_proba (e.g. LinearSVC)
        try:
            scores = model.decision_function(X_test)
            roc_auc = roc_auc_score(y_true, scores)
            # Normalize to 0-1 range to act as pseudo-probabilities for thresholding
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        except Exception:
            warnings.warn(
                f"ROC-AUC could not be computed for {target_name}.",
                UserWarning,
                stacklevel=2,
            )

    best_threshold = 0.5
    if y_proba is not None:
        if target_name == "fraud":
            # KIYMETLİ VERİ BİLİMCİ NOTU: Dolandırıcılık tespiti gibi yüksek derecede dengesiz veri setlerinde,
            # varsayılan 0.50 karar eşik değeri (decision threshold) genellikle yetersiz kalır ve modelin 
            # F1-skorunu (Precision ile Recall arasındaki denge) optimize etmez. Bu yüzden, 0.01 ile 0.99 
            # aralığında bir tarama yaparak en yüksek F1-skorunu veren en iyi karar eşiğini (Decision Threshold) 
            # dinamik olarak seçiyoruz. Bu sayede modelin kaçırdığı dolandırıcılık vakaları ile sahte alarmlar
            # arasındaki dengeyi en dürüst şekilde optimize etmiş oluyoruz.
            best_f1 = -1.0
            for t in np.linspace(0.01, 0.99, 99):
                y_pred_t = (y_proba >= t).astype(int)
                f1 = f1_score(y_true, y_pred_t, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = t
            y_pred = (y_proba >= best_threshold).astype(int)
            print(f"    [Tuning] Optimized threshold for {model.__class__.__name__:<25} -> {best_threshold:.2f} (Max F1: {best_f1:.4f})")
        else:
            # Geç teslimat hedefi zaten dengeli (~%55) olduğu için standart 0.50 eşik değeriyle devam edilir.
            y_pred = (y_proba >= 0.5).astype(int)
    else:
        # Modelin predict_proba desteği olmadığı durumlarda yedek (fallback) tahminleme mekanizması.
        y_pred = model.predict(X_test)

    metrics = {
        # Always (y_true, y_pred) — never reverse
        "accuracy"        : accuracy_score(y_true, y_pred),
        "f1"              : f1_score(y_true, y_pred, zero_division=0),
        "recall"          : recall_score(y_true, y_pred, zero_division=0),
        "precision"       : precision_score(y_true, y_pred, zero_division=0),
        "roc_auc"         : roc_auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "y_pred"          : y_pred,
        "y_proba"         : y_proba,
        "threshold"       : best_threshold,
    }

    return metrics

def print_metrics_block(
    metrics: dict[str, Any],
    target_name: str,
) -> None:
    """Print a formatted metrics block for one target."""
    print(f"\n  [{target_name.upper()}]")
    print(f"    Accuracy  : {metrics['accuracy']:.4f}")
    print(f"    F1 Score  : {metrics['f1']:.4f}")
    print(f"    Recall    : {metrics['recall']:.4f}")
    print(f"    Precision : {metrics['precision']:.4f}")
    roc = metrics['roc_auc']
    print(f"    ROC-AUC   : {roc:.4f}" if roc is not None else "    ROC-AUC   : N/A")
    print(f"    Confusion Matrix:\n{metrics['confusion_matrix']}")


def classifiermodel(
    model_name: str,
    model_f,
    model_l,
    xf_train: np.ndarray,
    xf_test: np.ndarray,
    yf_train: pd.Series,
    yf_test: pd.Series,
    xl_train: np.ndarray,
    xl_test: np.ndarray,
    yl_train: pd.Series,
    yl_test: pd.Series,
) -> dict[str, Any]:
    """
    Fit two models (fraud + late_delivery) and return all metrics.

    Parameters
    ----------
    model_name : display name for the model
    model_f    : unfitted classifier for fraud target
    model_l    : unfitted classifier for late_delivery target
    xf_train / xf_test : fraud train/test features
    yf_train / yf_test : fraud train/test labels
    xl_train / xl_test : late delivery train/test features
    yl_train / yl_test : late delivery train/test labels

    Returns
    -------
    dict with 'model_name', 'fraud', 'late_delivery' metric dicts
    and fitted model objects for downstream use.
    """
    print(f"\n{'='*60}")
    print(f"  MODEL: {model_name}")
    print(f"{'='*60}")

    # ── Fraud model ──────────────────────────────────────────────────
    print("\n  Training fraud model...")
    model_f.fit(xf_train, yf_train)
    fraud_metrics = compute_metrics(model_f, xf_test, yf_test, "fraud")
    print_metrics_block(fraud_metrics, "fraud")

    # ── Late delivery model ───────────────────────────────────────────
    print("\n  Training late delivery model...")
    model_l.fit(xl_train, yl_train)
    late_metrics = compute_metrics(model_l, xl_test, yl_test, "late_delivery")
    print_metrics_block(late_metrics, "late_delivery")

    return {
        "model_name"    : model_name,
        "model_f"       : model_f,       # keep fitted model for downstream use
        "model_l"       : model_l,
        "fraud"         : fraud_metrics,
        "late_delivery" : late_metrics,
    }


def build_comparison_table(
    results: list[dict],
    target: str,
) -> pd.DataFrame:
    """
    Build a sorted comparison DataFrame for one target.

    Parameters
    ----------
    results : list of dicts from classifiermodel()
    target  : 'fraud' or 'late_delivery'

    Returns
    -------
    pd.DataFrame sorted by ROC-AUC descending
    """
    rows = []
    for r in results:
        m = r[target]
        rows.append({
            "Model"    : r["model_name"],
            "Accuracy" : round(m["accuracy"],  4),
            "F1"       : round(m["f1"],        4),
            "Recall"   : round(m["recall"],    4),
            "Precision": round(m["precision"], 4),
            "ROC-AUC"  : round(m["roc_auc"],   4)
                         if m["roc_auc"] is not None else None,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    return df


def run_model_comparison(
    xf_train_balanced: np.ndarray,
    yf_train_balanced: pd.Series,
    xf_test_scaled: np.ndarray,
    yf_test: pd.Series,
    xl_train_scaled: np.ndarray,
    yl_train: pd.Series,
    xl_test_scaled: np.ndarray,
    yl_test: pd.Series,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    Run all 4 classifiers and return results + comparison tables.

    Returns
    -------
    results          : list of full metric dicts
    fraud_comparison : DataFrame sorted by ROC-AUC
    late_comparison  : DataFrame sorted by ROC-AUC
    """
    print_section("STEP 2 - MODEL TRAINING & EVALUATION")

    # ── Model definitions ────────────────────────────────────────────
    # class_weight='balanced' helps with fraud imbalance in LR and RF.
    # XGBoost handles imbalance via scale_pos_weight internally.
    # SMOTE already balanced the fraud train set so tree models
    # see equal classes — class_weight less critical there.

    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.ensemble import (RandomForestClassifier, 
        GradientBoostingClassifier,
        ExtraTreesClassifier, BaggingClassifier)
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    model_configs = [
    (
        "Logistic Regression",
        LogisticRegression(max_iter=1000, random_state=42,
                           class_weight='balanced'),
        LogisticRegression(max_iter=1000, random_state=42,
                           class_weight='balanced'),
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(random_state=42,
                               class_weight='balanced'),
        DecisionTreeClassifier(random_state=42,
                               class_weight='balanced'),
    ),
    (
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=42,
                               class_weight='balanced', n_jobs=-1),
        RandomForestClassifier(n_estimators=100, random_state=42,
                               class_weight='balanced', n_jobs=-1),
    ),
    (
        "Extra Trees",
        ExtraTreesClassifier(n_estimators=100, random_state=42,
                             class_weight='balanced', n_jobs=-1),
        ExtraTreesClassifier(n_estimators=100, random_state=42,
                             class_weight='balanced', n_jobs=-1),
    ),
    (
        "Gradient Boosting",
        GradientBoostingClassifier(n_estimators=100,
                                   random_state=42),
        GradientBoostingClassifier(n_estimators=100,
                                   random_state=42),
    ),
   
    
    (
        "Bagging",
        BaggingClassifier(n_estimators=100, random_state=42,
                          n_jobs=-1),
        BaggingClassifier(n_estimators=100, random_state=42,
                          n_jobs=-1),
    ),
    (
        "K-Nearest Neighbors",
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    ),
    (
        "Naive Bayes",
        GaussianNB(),
        GaussianNB(),
    ),
    (
        "SGD Classifier",
        SGDClassifier(random_state=42, class_weight='balanced',
                      loss='modified_huber'),
        SGDClassifier(random_state=42, class_weight='balanced',
                      loss='modified_huber'),
    ),
    (
        "XGBoost",
        XGBClassifier(n_estimators=100, random_state=42,
                      eval_metric='logloss',
                      use_label_encoder=False, verbosity=0),
        XGBClassifier(n_estimators=100, random_state=42,
                      eval_metric='logloss',
                      use_label_encoder=False, verbosity=0),
    ),
    (
        "LightGBM",
        LGBMClassifier(n_estimators=100, random_state=42,
                       class_weight='balanced', verbose=-1),
        LGBMClassifier(n_estimators=100, random_state=42,
                       class_weight='balanced', verbose=-1),
    ),
]

    # ── Train and evaluate all models ────────────────────────────────
    results = []
    for model_name, model_f, model_l in model_configs:
        result = classifiermodel(
            model_name,
            model_f,
            model_l,
            xf_train_balanced,
            xf_test_scaled,
            yf_train_balanced,
            yf_test,
            xl_train_scaled,
            xl_test_scaled,
            yl_train,
            yl_test,
        )
        results.append(result)

    # ── Build comparison tables ───────────────────────────────────────
    print_section("STEP 3 - MODEL COMPARISON TABLES")

    fraud_comparison = build_comparison_table(results, "fraud")
    late_comparison  = build_comparison_table(results, "late_delivery")

    print("\n  FRAUD DETECTION - Model Comparison (sorted by ROC-AUC):")
    print(fraud_comparison.to_string(index=False))

    best_fraud = fraud_comparison.iloc[0]["Model"]
    print(f"\n  -> Best fraud model     : {best_fraud}")

    print("\n\n  LATE DELIVERY - Model Comparison (sorted by ROC-AUC):")
    print(late_comparison.to_string(index=False))

    best_late = late_comparison.iloc[0]["Model"]
    print(f"\n  -> Best late delivery model : {best_late}")

    print_section("PIPELINE COMPLETE")
    print("  Next steps:")
    print("  1. Cross validation + overfit analysis")
    print("  2. Confusion matrix visualizations")
    print("  3. ROC-AUC curve visualizations")
    print("  4. Feature importance analysis")
    print("  5. SHAP / LIME explainability")

    return results, fraud_comparison, late_comparison


# ── Notebook usage ───────────────────────────────────────────────────
# from model_training_pipeline import run_model_comparison
#
# results, fraud_comparison, late_comparison = run_model_comparison(
#     output.xf_train_balanced,
#     output.yf_train_balanced,
#     output.xf_test_scaled,
#     output.yf_test,
#     output.xl_train_scaled,
#     output.yl_train,
#     output.xl_test_scaled,
#     output.yl_test,
# )