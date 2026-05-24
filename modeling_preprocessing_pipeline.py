"""
Train/test split, scaling, and class-imbalance handling pipeline.

This module prepares two independent binary-classification datasets:
    1. Fraud model
    2. Late-delivery model

Expected input:
    train_data
        Output of data_preparation_pipeline
        .prepare_training_data(df)
        Must be fully numeric (LabelEncoder applied).
        Expected feature count: ~20-30 columns.

Encoding assumption:
    data_preparation_pipeline uses LabelEncoder for all
    categorical columns. This pipeline does NOT re-encode.
    label_encoders.pkl must exist for inference reuse.

Missing value handling:
    Median imputation is applied after train/test split
    to prevent leakage. With LabelEncoder preparation,
    missing values are rare but imputer is retained for
    inference-time robustness.

Final outputs ready for model training:
    FRAUD MODEL
        xf_train_balanced  (SMOTE balanced — for model training)
        yf_train_balanced
        xf_test_scaled     (original distribution)
        yf_test
        xf_train_scaled    (pre-SMOTE — for CV fold-level SMOTE)
        yf_train           (original imbalanced — for CV)

    LATE DELIVERY MODEL
        xl_train_scaled
        yl_train
        xl_test_scaled
        yl_test
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


TEST_SIZE    = 0.2
RANDOM_STATE = 42
SMOTE_K      = 5

SCALER_FRAUD_PATH         = Path("scaler_fraud.pkl")
SCALER_LATE_DELIVERY_PATH = Path("scaler_late_delivery.pkl")
IMPUTER_FRAUD_PATH        = Path("imputer_fraud.pkl")
IMPUTER_LATE_DELIVERY_PATH= Path("imputer_late_delivery.pkl")


@dataclass
class ModelingPreprocessingOutput:
    """Container for all arrays/series needed by downstream model training."""

    # ── Fraud model ───────────────────────────────────────────────
    xf_train_balanced: object    # SMOTE balanced  — use for model training
    yf_train_balanced: pd.Series
    xf_test_scaled:    object    # original dist   — use for final evaluation
    yf_test:           pd.Series

    # Pre-SMOTE fraud train — use for CV (SMOTE applied inside each fold)
    xf_train_scaled:   object
    yf_train:          pd.Series

    # ── Late delivery model ───────────────────────────────────────
    xl_train_scaled:   object
    yl_train:          pd.Series
    xl_test_scaled:    object
    yl_test:           pd.Series

    # ── Fitted preprocessors ─────────────────────────────────────
    scaler_f:  RobustScaler
    scaler_l:  RobustScaler
    imputer_f: SimpleImputer
    imputer_l: SimpleImputer


def print_section(title: str) -> None:
    """Print clear stage boundaries for notebook and log readability."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def check_label_encoders_exist() -> None:
    """Warn if label_encoders.pkl is missing."""
    encoder_path = Path("label_encoders.pkl")
    if encoder_path.exists():
        print(f"  label_encoders.pkl found at {encoder_path.resolve()}")
    else:
        warnings.warn(
            "label_encoders.pkl not found. "
            "Run data_preparation_pipeline.prepare_training_data(df) first. "
            "Inference on new data will fail without fitted encoders.",
            UserWarning,
            stacklevel=2,
        )


def print_class_distribution(y: pd.Series, target_name: str) -> None:
    """Print count and percentage distribution for a binary target."""
    counts      = y.value_counts(dropna=False).sort_index()
    percentages = y.value_counts(normalize=True, dropna=False).sort_index() * 100
    summary = pd.DataFrame({
        "count"           : counts,
        "class_balance_pct": percentages.round(2),
    })
    print(f"\n{target_name} class distribution:")
    print(summary)


def validate_train_data(train_data: pd.DataFrame) -> None:
    """Validate required modeling columns before any split/scaling work starts."""
    required_targets = ["fraud", "late_delivery"]
    missing_targets  = [c for c in required_targets
                        if c not in train_data.columns]
    if missing_targets:
        raise KeyError(
            f"train_data is missing required target columns: {missing_targets}")

    non_numeric = train_data.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        warnings.warn(
            f"Non-numeric columns detected: {non_numeric}. "
            "Ensure data_preparation_pipeline ran successfully.",
            UserWarning, stacklevel=2)
    else:
        print("  Validation passed: all columns are numeric.")

    feature_cols = [c for c in train_data.columns
                    if c not in ("fraud", "late_delivery")]
    if len(feature_cols) > 100:
        warnings.warn(
            f"Feature count is {len(feature_cols)}, which suggests "
            "OneHotEncoder may have been applied instead of LabelEncoder. "
            "Expected ~20-30 features after LabelEncoder-based preparation.",
            UserWarning, stacklevel=2)
    else:
        print(f"  Feature count: {len(feature_cols)} — consistent "
              "with LabelEncoder preparation.")


def separate_targets(
    train_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Step 1: Create independent feature/target pairs for each model."""
    print_section("STEP 1 - TARGET SEPARATION")
    validate_train_data(train_data)

    xf = train_data.drop(columns=["fraud", "late_delivery"])
    yf = train_data["fraud"]
    xl = train_data.drop(columns=["late_delivery", "fraud"])
    yl = train_data["late_delivery"]

    print(f"Fraud features shape      : {xf.shape}")
    print_class_distribution(yf, "fraud")
    print(f"\nLate delivery features shape: {xl.shape}")
    print_class_distribution(yl, "late_delivery")

    return xf, yf, xl, yl


def split_train_test(
    xf: pd.DataFrame, yf: pd.Series,
    xl: pd.DataFrame, yl: pd.Series,
) -> tuple:
    """Step 2: Stratified train/test split for both targets."""
    print_section("STEP 2 - TRAIN/TEST SPLIT")

    try:
        xf_train, xf_test, yf_train, yf_test = train_test_split(
            xf, yf, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=yf)

        xl_train, xl_test, yl_train, yl_test = train_test_split(
            xl, yl, test_size=TEST_SIZE,
            random_state=RANDOM_STATE, stratify=yl)
    except ValueError as exc:
        raise ValueError(f"Train/test split failed: {exc}") from exc

    print("Fraud split shapes:")
    print(f"  xf_train: {xf_train.shape}  |  xf_test: {xf_test.shape}")
    print_class_distribution(yf_train, "fraud train")
    print_class_distribution(yf_test,  "fraud test")

    print("\nLate delivery split shapes:")
    print(f"  xl_train: {xl_train.shape}  |  xl_test: {xl_test.shape}")
    print_class_distribution(yl_train, "late_delivery train")
    print_class_distribution(yl_test,  "late_delivery test")

    return (xf_train, xf_test, yf_train, yf_test,
            xl_train, xl_test, yl_train, yl_test)


def impute_missing_values(
    xf_train: pd.DataFrame, xf_test: pd.DataFrame,
    xl_train: pd.DataFrame, xl_test: pd.DataFrame,
) -> tuple:
    """
    Impute numeric missing values using medians learned from train only.

    With LabelEncoder preparation missing values are rare but imputer
    is retained for inference-time robustness.
    """
    print_section("STEP 3A - MISSING VALUE IMPUTATION")

    print("Missing values before imputation:")
    print(f"  Fraud train        : {int(xf_train.isna().sum().sum()):,}")
    print(f"  Fraud test         : {int(xf_test.isna().sum().sum()):,}")
    print(f"  Late delivery train: {int(xl_train.isna().sum().sum()):,}")
    print(f"  Late delivery test : {int(xl_test.isna().sum().sum()):,}")

    imputer_f = SimpleImputer(strategy="median")
    imputer_l = SimpleImputer(strategy="median")

    try:
        xf_train_imp = imputer_f.fit_transform(xf_train)
        xf_test_imp  = imputer_f.transform(xf_test)
        xl_train_imp = imputer_l.fit_transform(xl_train)
        xl_test_imp  = imputer_l.transform(xl_test)
    except Exception as exc:
        raise RuntimeError(f"Imputation failed: {exc}") from exc

    print("\n  Strategy         : median")
    print("  Leakage control  : fit on train, transform test")

    return (xf_train_imp, xf_test_imp,
            xl_train_imp, xl_test_imp,
            imputer_f, imputer_l)


def scale_features(
    xf_train_imp: object, xf_test_imp: object,
    xl_train_imp: object, xl_test_imp: object,
) -> tuple:
    """Step 3B: Fit RobustScaler on train only, then transform train/test."""
    print_section("STEP 3B - SCALING WITH ROBUSTSCALER")

    scaler_f = RobustScaler()
    scaler_l = RobustScaler()

    try:
        xf_train_scaled = scaler_f.fit_transform(xf_train_imp)
        xf_test_scaled  = scaler_f.transform(xf_test_imp)
        xl_train_scaled = scaler_l.fit_transform(xl_train_imp)
        xl_test_scaled  = scaler_l.transform(xl_test_imp)
    except Exception as exc:
        raise RuntimeError(f"Scaling failed: {exc}") from exc

    print("  Fraud scaler : fit on xf_train — RobustScaler (outlier-safe)")
    print("  Late scaler  : fit on xl_train — RobustScaler (outlier-safe)")
    print("  Leakage ctrl : no scaler fitted on test data")

    return (xf_train_scaled, xf_test_scaled,
            xl_train_scaled, xl_test_scaled,
            scaler_f, scaler_l)


def handle_class_imbalance(
    xf_train_scaled: object,
    yf_train: pd.Series,
    xl_train_scaled: object,
    yl_train: pd.Series,
) -> tuple[object, pd.Series]:
    """
    Step 4: Apply SMOTE only to the fraud training set.

    xf_train_scaled (pre-SMOTE) is kept in the output object
    for CV pipelines that apply SMOTE inside each fold.
    """
    print_section("STEP 4 - CLASS IMBALANCE HANDLING")

    print("Fraud train distribution BEFORE SMOTE:")
    print_class_distribution(yf_train, "fraud train")

    try:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
        xf_train_balanced, yf_train_balanced = smote.fit_resample(
            xf_train_scaled, yf_train)
    except ValueError as exc:
        raise ValueError(f"SMOTE failed: {exc}") from exc

    print("\nFraud train distribution AFTER SMOTE:")
    print_class_distribution(
        pd.Series(yf_train_balanced), "fraud train balanced")
    print(f"  xf_train_balanced shape: {xf_train_balanced.shape}")

    print("\nLate delivery — NO SMOTE (balanced ~55%)")
    print_class_distribution(yl_train, "late_delivery train")

    return xf_train_balanced, yf_train_balanced


def save_preprocessors(
    scaler_f:  RobustScaler,
    scaler_l:  RobustScaler,
    imputer_f: SimpleImputer,
    imputer_l: SimpleImputer,
) -> None:
    """Step 5: Persist fitted imputers and scalers for inference-time reuse."""
    print_section("STEP 5 - SAVE PREPROCESSORS")

    try:
        joblib.dump(imputer_f, IMPUTER_FRAUD_PATH)
        joblib.dump(imputer_l, IMPUTER_LATE_DELIVERY_PATH)
        joblib.dump(scaler_f,  SCALER_FRAUD_PATH)
        joblib.dump(scaler_l,  SCALER_LATE_DELIVERY_PATH)
    except Exception as exc:
        raise OSError(f"Could not save preprocessors: {exc}") from exc

    print(f"  Saved: {IMPUTER_FRAUD_PATH}")
    print(f"  Saved: {IMPUTER_LATE_DELIVERY_PATH}")
    print(f"  Saved: {SCALER_FRAUD_PATH}")
    print(f"  Saved: {SCALER_LATE_DELIVERY_PATH}")


def print_final_summary(
    xf_train_balanced: object,
    xf_train_scaled:   object,
    xf_test_scaled:    object,
    xl_train_scaled:   object,
    xl_test_scaled:    object,
) -> None:
    """Step 6: Print concise preprocessing summary."""
    print_section("STEP 6 - FINAL SUMMARY")

    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Fraud train (SMOTE balanced) : {xf_train_balanced.shape}")
    print(f"Fraud train (pre-SMOTE / CV) : {xf_train_scaled.shape}")
    print(f"Fraud test  (original dist)  : {xf_test_scaled.shape}")
    print(f"Late delivery train          : {xl_train_scaled.shape}")
    print(f"Late delivery test           : {xl_test_scaled.shape}")
    print(f"Feature count                : {xf_train_scaled.shape[1]}")
    print("Scaler                       : RobustScaler")
    print("Imputer                      : median")
    print("Fraud imbalance fix          : SMOTE (train only)")
    print("Late delivery imbalance      : None (balanced)")
    print("─" * 60)
    print("CV Usage:")
    print("  output.xf_train_scaled  ← SMOTE applied inside each fold")
    print("  output.yf_train         ← original imbalanced labels")
    print("Model Training Usage:")
    print("  output.xf_train_balanced ← pre-balanced by SMOTE")
    print("=" * 60)


def run_modeling_preprocessing(
    train_data: pd.DataFrame,
) -> ModelingPreprocessingOutput:
    """
    Run the complete modeling-preprocessing pipeline.

    Parameters
    ----------
    train_data : pd.DataFrame
        Numeric prepared dataset from data_preparation_pipeline.
        Must contain 'fraud' and 'late_delivery' target columns.

    Returns
    -------
    ModelingPreprocessingOutput
        All train/test arrays, balanced + pre-SMOTE fraud data,
        labels, imputers, and scalers.
    """
    check_label_encoders_exist()

    xf, yf, xl, yl = separate_targets(train_data)

    (xf_train, xf_test, yf_train, yf_test,
     xl_train, xl_test, yl_train, yl_test) = split_train_test(
        xf, yf, xl, yl)

    (xf_train_imp, xf_test_imp,
     xl_train_imp, xl_test_imp,
     imputer_f, imputer_l) = impute_missing_values(
        xf_train, xf_test, xl_train, xl_test)

    (xf_train_scaled, xf_test_scaled,
     xl_train_scaled, xl_test_scaled,
     scaler_f, scaler_l) = scale_features(
        xf_train_imp, xf_test_imp,
        xl_train_imp, xl_test_imp)

    xf_train_balanced, yf_train_balanced = handle_class_imbalance(
        xf_train_scaled, yf_train,
        xl_train_scaled, yl_train)

    save_preprocessors(scaler_f, scaler_l, imputer_f, imputer_l)

    print_final_summary(
        xf_train_balanced, xf_train_scaled,
        xf_test_scaled, xl_train_scaled, xl_test_scaled)

    return ModelingPreprocessingOutput(
        # Fraud — model training
        xf_train_balanced = xf_train_balanced,
        yf_train_balanced = yf_train_balanced,
        xf_test_scaled    = xf_test_scaled,
        yf_test           = yf_test,
        # Fraud — CV (pre-SMOTE)
        xf_train_scaled   = xf_train_scaled,
        yf_train          = yf_train,
        # Late delivery
        xl_train_scaled   = xl_train_scaled,
        yl_train          = yl_train,
        xl_test_scaled    = xl_test_scaled,
        yl_test           = yl_test,
        # Preprocessors
        scaler_f          = scaler_f,
        scaler_l          = scaler_l,
        imputer_f         = imputer_f,
        imputer_l         = imputer_l,
    )


# ══════════════════════════════════════════════════════════════════
#  NOTEBOOK USAGE
# ══════════════════════════════════════════════════════════════════
# from modeling_preprocessing_pipeline import run_modeling_preprocessing
#
# output = run_modeling_preprocessing(train_data)
#
# # Model training:
# output.xf_train_balanced, output.yf_train_balanced
# output.xf_test_scaled,    output.yf_test
#
# # CV (SMOTE inside fold):
# output.xf_train_scaled,   output.yf_train
