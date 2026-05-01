import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
import seaborn as sns


LABEL_ORDER = ["negative", "neutral", "positive"]
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def build_error_df(test_df, y_pred, proba=None, id2label=ID2LABEL):
    """
    Builds dataframe with text, true label, predicted label, confidence and error flag.
    """

    out = test_df.copy().reset_index(drop=True)

    out["true_label"] = out["label_id"].map(id2label)
    out["pred_id"] = y_pred
    out["pred_label"] = out["pred_id"].map(id2label)

    out["is_error"] = out["true_label"] != out["pred_label"]

    if proba is not None:
        out["confidence"] = proba.max(axis=1)
        for i, label in id2label.items():
            out[f"proba_{label}"] = proba[:, i]
    else:
        out["confidence"] = np.nan

    return out


def evaluate_error_df(error_df, label_order=LABEL_ORDER):
    """
    Prints metrics from an error dataframe.
    """

    y_true = error_df["true_label"]
    y_pred = error_df["pred_label"]

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print("Accuracy:", acc)
    print("Macro F1:", f1_macro)
    print("Weighted F1:", f1_weighted)
    print()
    print(classification_report(
        y_true,
        y_pred,
        labels=label_order,
        zero_division=0
    ))

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }


def plot_confusion(error_df, title="Confusion Matrix", label_order=LABEL_ORDER):
    cm = confusion_matrix(
        error_df["true_label"],
        error_df["pred_label"],
        labels=label_order
    )

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_order,
        yticklabels=label_order
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def get_errors(error_df, n=20, highest_confidence=True):
    """
    Shows wrong predictions.
    """

    cols = ["text", "true_label", "pred_label", "confidence"]
    proba_cols = [c for c in error_df.columns if c.startswith("proba_")]
    cols += proba_cols

    errors = error_df[error_df["is_error"]].copy()

    if highest_confidence:
        errors = errors.sort_values("confidence", ascending=False)

    return errors[cols].head(n)


def compare_errors(error_df_a, error_df_b, name_a="model_a", name_b="model_b"):
    """
    Compare two models on the same test set.
    """

    comp = pd.DataFrame({
        "text": error_df_a["text"].values,
        "true_label": error_df_a["true_label"].values,
        f"pred_{name_a}": error_df_a["pred_label"].values,
        f"error_{name_a}": error_df_a["is_error"].values,
        f"pred_{name_b}": error_df_b["pred_label"].values,
        f"error_{name_b}": error_df_b["is_error"].values,
    })

    comp[f"error_{name_a}"] = comp[f"error_{name_a}"].astype(bool)
    comp[f"error_{name_b}"] = comp[f"error_{name_b}"].astype(bool)

    comp["case"] = np.select(
        [
            (comp[f"error_{name_a}"]) & (~comp[f"error_{name_b}"]),
            (~comp[f"error_{name_a}"]) & (comp[f"error_{name_b}"]),
            (comp[f"error_{name_a}"]) & (comp[f"error_{name_b}"]),
            (~comp[f"error_{name_a}"]) & (~comp[f"error_{name_b}"]),
        ],
        [
            f"Only {name_a} wrong",
            f"Only {name_b} wrong",
            "Both wrong",
            "Both correct",
        ],
        default="Unknown"
    )


    return comp
