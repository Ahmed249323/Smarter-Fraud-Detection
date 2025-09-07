import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, average_precision_score,
                           precision_recall_curve, roc_curve,
                           accuracy_score, balanced_accuracy_score,
                           f1_score, precision_score, recall_score,
                           log_loss, matthews_corrcoef)

from sklearn.metrics import classification_report
import joblib
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


def load_csv(path: str):
    """
    Load a CSV file into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    df : DataFrame
        Loaded dataset.
    """
    df = pd.read_csv(path)
    return df


def split_xy(df, target_col: str):
    """
    Split a DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : DataFrame
        Input dataset.
    target_col : str
        Name of the target column.

    Returns
    -------
    X : DataFrame
        Features (all columns except target).
    y : Series
        Target column.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def check_missing_and_duplicates(df: pd.DataFrame, drop_duplicates: bool = False):
    """
    Check for missing values and duplicate rows in a DataFrame.
    Optionally drop duplicate rows.

    Parameters
    ----------
    df : DataFrame
        Input dataset.
    drop_duplicates : bool, default=False
        Whether to drop duplicate rows.

    Returns
    -------
    report : dict
        Dictionary with missing values summary and duplicate count.
    df_clean : DataFrame
        The (optionally) cleaned DataFrame.
    """
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

    duplicate_count = df.duplicated().sum()

    if drop_duplicates and duplicate_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)

    report = {
        "missing_values": missing_summary,
        "duplicate_rows": duplicate_count,
        "duplicates_dropped": drop_duplicates
    }

    return report, df


def train_on_train_with_threshold(X_train, y_train, model, save_path=None, threshold=0.5):
    """
    Train a classifier on the training set and apply a probability threshold
    to the training predictions only.

    Parameters
    ----------
    X_train, y_train : training data
    model : sklearn-like classifier
    save_path : str, optional
        Path to save the trained model.
    threshold : float, default=0.5
        Probability threshold for positive class.

    Returns
    -------
    model : trained model
    report : dict
        Classification metrics on thresholded training predictions
    y_train_thresh : array
        Thresholded training predictions (0/1)
    """
    model.fit(X_train, y_train)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_train)[:, 1] 
        y_train_thresh = (y_proba >= threshold).astype(int)
    else:
        y_train_thresh = model.predict(X_train)
    report = classification_report(y_train, y_train_thresh, output_dict=True)
    print(f"✅ Training set evaluation with threshold={threshold}:")
    # print(classification_report(y_train, y_train_thresh))

    if save_path:
        joblib.dump(model, save_path)
        print(f"✅ Model saved at: {save_path}")

    return model, report, y_train_thresh


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE oversampling to balance the training set.

    Parameters
    ----------
    X_train : DataFrame
        Training features.
    y_train : Series
        Training target.
    random_state : int, default=42
        Random state for reproducibility.

    Returns
    -------
    X_resampled : DataFrame
        Oversampled training features.
    y_resampled : Series
        Oversampled training target.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled




def evaluate_binary_model(model, X, y, threshold=0.5):
    """Extended evaluation for binary imbalanced datasets"""
    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        'accuracy': accuracy_score(y, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y, y_pred),
        'f1_score': f1_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y, probs),
        'pr_auc': average_precision_score(y, probs),
        'log_loss': log_loss(y, probs),
        'mcc': matthews_corrcoef(y, y_pred),
        'specificity': tn / (tn + fp + 1e-8),
        'npv': tn / (tn + fn + 1e-8)
    }

def plot_binary_model_curves(model, X, y, threshold=0.5, title_prefix="Model"):
    """Plot ROC curve, PR curve, and Confusion Matrix for binary classification"""
    probs = model.predict_proba(X)[:, 1]
    y_pred = (probs >= threshold).astype(int)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title(f"{title_prefix} - ROC Curve")
    plt.legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y, probs)
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, label="PR Curve", color="orange")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title_prefix} - Precision-Recall Curve")
    plt.legend()

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.subplot(1, 3, 3)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{title_prefix} - Confusion Matrix")

    plt.tight_layout()
    plt.show()

