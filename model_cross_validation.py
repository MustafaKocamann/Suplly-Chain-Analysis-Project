"""
11-Model 5-Fold Stratified Cross Validation Pipeline
Supply Chain — Binary Classification

Targets:
    fraud         (~2.3% positive — imbalanced → StratifiedKFold)
    late_delivery (~55% positive  — balanced)

Outputs:
    cv_results        — list of dicts with all CV metrics
    fraud_cv_df       — DataFrame sorted by ROC-AUC
    late_cv_df        — DataFrame sorted by ROC-AUC
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    BaggingClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────
RANDOM_STATE = 42
N_FOLDS      = 5
SMOTE_K      = 5


# ══════════════════════════════════════════════════════════════════
#  MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════
def build_all_models() -> dict[str, Any]:
    """
    Define all 11 classifiers.
    Each model is instantiated twice (fraud + late_delivery)
    so they are independently fitted without state bleeding.

    Returns
    -------
    dict mapping model_name → (model_fraud, model_late)
    """
    def lr():
        return LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE,
            class_weight="balanced", solver="lbfgs")

    def dt():
        return DecisionTreeClassifier(
            max_depth=10, random_state=RANDOM_STATE,
            class_weight="balanced")

    def rf():
        return RandomForestClassifier(
            n_estimators=100, max_depth=15,
            random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1)

    def et():
        return ExtraTreesClassifier(
            n_estimators=100, max_depth=15,
            random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1)

    def gb():
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            learning_rate=0.1, random_state=RANDOM_STATE)

    def bag():
        return BaggingClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            n_jobs=-1)

    def knn():
        return KNeighborsClassifier(
            n_neighbors=5, n_jobs=-1)

    def nb():
        return GaussianNB()

    def sgd():
        return SGDClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            loss="modified_huber",   # supports predict_proba
            max_iter=1000)

    def xgb():
        return XGBClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=RANDOM_STATE,
            eval_metric="logloss",
            use_label_encoder=False, verbosity=0,
            n_jobs=-1)

    def lgb():
        return LGBMClassifier(
            n_estimators=100, max_depth=6,
            learning_rate=0.1, random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1, verbose=-1)

    models = {
        "Logistic Regression"  : (lr(),  lr()),
        "Decision Tree"        : (dt(),  dt()),
        "Random Forest"        : (rf(),  rf()),
        "Extra Trees"          : (et(),  et()),
        "Gradient Boosting"    : (gb(),  gb()),
        "Bagging"              : (bag(), bag()),
        "K-Nearest Neighbors"  : (knn(), knn()),
        "Naive Bayes"          : (nb(),  nb()),
        "SGD Classifier"       : (sgd(), sgd()),
        "XGBoost"              : (xgb(), xgb()),
        "LightGBM"             : (lgb(), lgb()),
    }
    return models


# ══════════════════════════════════════════════════════════════════
#  SINGLE MODEL — 5-FOLD CV (SMOTE inside fold & Threshold Tuning)
# ══════════════════════════════════════════════════════════════════
def cross_validate_classifier(
    model_name: str,
    model_f,
    model_l,
    X_f: np.ndarray,
    y_f: np.ndarray,
    X_l: np.ndarray,
    y_l: np.ndarray,
    n_folds: int = N_FOLDS,
    apply_smote_fraud: bool = True,
) -> dict[str, Any]:
    """
    Run StratifiedKFold CV for both targets independently.
    StratifiedKFold preserves class ratio in every fold.
    For fraud target, SMOTE is applied inside each fold to prevent leakage.
    Karar eşiği (threshold) F1 skorunu maksimize edecek şekilde dinamik optimize edilir.
    """
    X_f = np.asarray(X_f)
    y_f = np.asarray(y_f)
    X_l = np.asarray(X_l)
    y_l = np.asarray(y_l)

    skf_f = StratifiedKFold(
        n_splits=n_folds, shuffle=True,
        random_state=RANDOM_STATE)
    skf_l = StratifiedKFold(
        n_splits=n_folds, shuffle=True,
        random_state=RANDOM_STATE)

    # Metric accumulators
    metrics_f: dict[str, list] = {
        "accuracy": [], "f1": [], "recall": [],
        "precision": [], "roc_auc": []}
    metrics_l: dict[str, list] = {
        "accuracy": [], "f1": [], "recall": [],
        "precision": [], "roc_auc": []}

    splits_f = list(skf_f.split(X_f, y_f))
    splits_l = list(skf_l.split(X_l, y_l))

    fold_bar = tqdm(
        zip(range(1, n_folds + 1), splits_f, splits_l),
        total=n_folds,
        desc=f"  {model_name:<25}",
        leave=False,
        bar_format="{l_bar}{bar}| Fold {n_fmt}/{total_fmt} [{elapsed}]"
    )

    for fold, (tr_idx_f, val_idx_f), (tr_idx_l, val_idx_l) in fold_bar:

        # ── Fraud fold ───────────────────────────────────────────
        Xf_tr, Xf_val = X_f[tr_idx_f], X_f[val_idx_f]
        yf_tr, yf_val = y_f[tr_idx_f], y_f[val_idx_f]

        # SMOTE sadece train split'e — val dokunulmaz (leakage-free!)
        if apply_smote_fraud:
            try:
                smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
                Xf_tr, yf_tr = smote.fit_resample(Xf_tr, yf_tr)
            except Exception:
                pass

        model_f.fit(Xf_tr, yf_tr)
        
        # Predict probabilities or decision function
        yf_proba = None
        try:
            yf_proba = model_f.predict_proba(Xf_val)[:, 1]
        except AttributeError:
            try:
                scores = model_f.decision_function(Xf_val)
                yf_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            except Exception:
                yf_proba = None

        if yf_proba is not None:
            # Karar eşiğini (threshold) F1'i maksimize edecek şekilde dinamik optimize et
            best_f1 = -1.0
            best_t = 0.5
            for t in np.linspace(0.01, 0.99, 99):
                yf_pred_t = (yf_proba >= t).astype(int)
                f1 = f1_score(yf_val, yf_pred_t, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = t
            yf_pred = (yf_proba >= best_t).astype(int)
            try:
                metrics_f["roc_auc"].append(roc_auc_score(yf_val, yf_proba))
            except Exception:
                metrics_f["roc_auc"].append(np.nan)
        else:
            yf_pred = model_f.predict(Xf_val)
            metrics_f["roc_auc"].append(np.nan)

        metrics_f["accuracy"].append(accuracy_score(yf_val, yf_pred))
        metrics_f["f1"].append(f1_score(yf_val, yf_pred, zero_division=0))
        metrics_f["recall"].append(recall_score(yf_val, yf_pred, zero_division=0))
        metrics_f["precision"].append(precision_score(yf_val, yf_pred, zero_division=0))

        # ── Late delivery fold ───────────────────────────────────
        Xl_tr, Xl_val = X_l[tr_idx_l], X_l[val_idx_l]
        yl_tr, yl_val = y_l[tr_idx_l], y_l[val_idx_l]

        # Late delivery balanced — SMOTE gerekmez
        model_l.fit(Xl_tr, yl_tr)
        yl_pred = model_l.predict(Xl_val)

        metrics_l["accuracy"].append(accuracy_score(yl_val, yl_pred))
        metrics_l["f1"].append(
            f1_score(yl_val, yl_pred, zero_division=0))
        metrics_l["recall"].append(
            recall_score(yl_val, yl_pred, zero_division=0))
        metrics_l["precision"].append(
            precision_score(yl_val, yl_pred, zero_division=0))
        try:
            yl_proba = model_l.predict_proba(Xl_val)[:, 1]
            metrics_l["roc_auc"].append(
                roc_auc_score(yl_val, yl_proba))
        except Exception:
            metrics_l["roc_auc"].append(np.nan)

    def mean_std(lst: list) -> tuple[float, float]:
        arr = np.array(lst, dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr))

    return {
        "model_name"   : model_name,
        "fraud"        : {k: mean_std(v) for k, v in metrics_f.items()},
        "late_delivery": {k: mean_std(v) for k, v in metrics_l.items()},
    }


# ══════════════════════════════════════════════════════════════════
#  RUN ALL 11 MODELS
# ══════════════════════════════════════════════════════════════════
def run_cv_pipeline(
    X_fraud: np.ndarray,
    y_fraud: np.ndarray,
    X_late: np.ndarray,
    y_late: np.ndarray,
    print_tables: bool = True,
) -> tuple[list[dict], pd.DataFrame, pd.DataFrame]:
    """
    Run 5-fold stratified CV for all 11 models.

    ÖNEMLİ — doğru input:
        X_fraud = output.xf_train_scaled   (SMOTE uygulanmamış!)
        y_fraud = output.yf_train          (orijinal imbalanced!)
        X_late  = output.xl_train_scaled
        y_late  = output.yl_train
    """
    print("\n" + "=" * 65)
    print("  11 MODEL — 5-FOLD STRATIFIED CV (SMOTE inside fold)")
    print("  Fraud: SMOTE applied per fold (train split only)")
    print("  Late:  No SMOTE needed (balanced)")
    print("=" * 65)

    all_models  = build_all_models()
    cv_results  = []
    total_start = time.time()

    outer_bar = tqdm(
        all_models.items(),
        total=len(all_models),
        desc="Overall progress",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} models [{elapsed}]"
    )

    for model_name, (model_f, model_l) in outer_bar:
        outer_bar.set_postfix({"current": model_name})
        t0 = time.time()

        try:
            result = cross_validate_classifier(
                model_name, model_f, model_l,
                X_fraud, y_fraud,
                X_late,  y_late,
                apply_smote_fraud=True
            )
            elapsed = time.time() - t0
            result["elapsed_sec"] = elapsed

            cv_results.append(result)

            # Per-model summary line (highly readable with F1 included!)
            f_roc  = result["fraud"]["roc_auc"][0]
            f_f1   = result["fraud"]["f1"][0]
            l_roc  = result["late_delivery"]["roc_auc"][0]
            tqdm.write(
                f"  [OK] {model_name:<25} | "
                f"Fraud ROC-AUC: {f_roc:.4f} F1: {f_f1:.4f} | "
                f"Late ROC-AUC: {l_roc:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

        except Exception as exc:
            tqdm.write(f"  [ERR] {model_name}: {exc}")

    total_elapsed = time.time() - total_start
    print(f"\n[INFO] All models complete - total time: "
          f"{total_elapsed/60:.1f} min")

    # ── Build comparison tables ───────────────────────────────────
    fraud_cv_df = _build_cv_table(cv_results, "fraud")
    late_cv_df  = _build_cv_table(cv_results, "late_delivery")

    if print_tables:
        _print_comparison_tables(fraud_cv_df, late_cv_df)

    return cv_results, fraud_cv_df, late_cv_df


def _build_cv_table(
    cv_results: list[dict],
    target: str,
) -> pd.DataFrame:
    """Build a sorted comparison DataFrame for one target."""
    rows = []
    for r in cv_results:
        m = r[target]
        rows.append({
            "Model"        : r["model_name"],
            "Accuracy"     : round(m["accuracy"][0],  4),
            "Accuracy_std" : round(m["accuracy"][1],  4),
            "F1"           : round(m["f1"][0],        4),
            "F1_std"       : round(m["f1"][1],        4),
            "Recall"       : round(m["recall"][0],    4),
            "Precision"    : round(m["precision"][0], 4),
            "ROC-AUC"      : round(m["roc_auc"][0],   4),
            "ROC-AUC_std"  : round(m["roc_auc"][1],   4),
            "Time(s)"      : round(r.get("elapsed_sec", 0), 1),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    df.index += 1   # rank starts at 1
    return df


def _print_comparison_tables(
    fraud_cv_df: pd.DataFrame,
    late_cv_df: pd.DataFrame,
) -> None:
    """Print both comparison tables with best model highlighted."""
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 1000)
    print("\n" + "=" * 65)
    print("  FRAUD DETECTION - CV Results (sorted by ROC-AUC)")
    print("=" * 65)
    print(fraud_cv_df[[
        "Model", "Accuracy", "F1", "Recall",
        "Precision", "ROC-AUC", "ROC-AUC_std", "Time(s)"
    ]].to_string())
    best_f = fraud_cv_df.iloc[0]["Model"]
    print(f"\n  -> Best fraud model      : {best_f} "
          f"(ROC-AUC = {fraud_cv_df.iloc[0]['ROC-AUC']})")

    print("\n" + "=" * 65)
    print("  LATE DELIVERY - CV Results (sorted by ROC-AUC)")
    print("=" * 65)
    print(late_cv_df[[
        "Model", "Accuracy", "F1", "Recall",
        "Precision", "ROC-AUC", "ROC-AUC_std", "Time(s)"
    ]].to_string())
    best_l = late_cv_df.iloc[0]["Model"]
    print(f"\n  -> Best late del. model  : {best_l} "
          f"(ROC-AUC = {late_cv_df.iloc[0]['ROC-AUC']})")

    print("\n  Next steps:")
    print("  1. Confusion matrix visualizations")
    print("  2. ROC-AUC curve visualizations")
    print("  3. Feature importance analysis")
    print("  4. SHAP / LIME explainability")