import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)



LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


def prepare_labels(df, text_col="headline", label_col="label", label2id=LABEL2ID):
    df_model = df[[text_col, label_col, 'n_other']].copy()
    df_model = df_model.rename(columns={text_col: "text", label_col: "label"})
    df_model["label_id"] = df_model["label"].map(label2id)
    df_model = df_model.dropna(subset=["text", "label_id"])
    df_model["label_id"] = df_model["label_id"].astype(int)
    return df_model


def random_split(df, label_col="label_id", test_size=0.2, val_size=0.1, random_state=42):
    train_df, temp_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    relative_test_size = test_size / (test_size + val_size)

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=temp_df[label_col],
    )

    return train_df, val_df, test_df


def entity_split(df, entity_count_col="n_entities", train_on="single", test_on="multiple"):

    if train_on == "single":
        train_df = df[df[entity_count_col] == 1].copy()
    elif train_on == "multiple":
        train_df = df[df[entity_count_col] > 1].copy()
    else:
        raise ValueError("train_on must be 'single' or 'multiple'.")

    if test_on == "single":
        test_df = df[df[entity_count_col] == 1].copy()
    elif test_on == "multiple":
        test_df = df[df[entity_count_col] > 1].copy()
    else:
        raise ValueError("test_on must be 'single' or 'multiple'.")

    return train_df, test_df


def evaluate_predictions(y_true, y_pred, target_names=None):

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    results = {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    print("Accuracy:", acc)
    print("Macro F1:", f1_macro)
    print("Weighted F1:", f1_weighted)
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

    return results


def train_logistic_regression(X_train, y_train, C=1.0, max_iter=2000, class_weight=None):
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight=class_weight,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
):
 
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y_train)),
        eval_metric="mlogloss",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def predict_classical(model, X):
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    else:
        proba = None

    return y_pred, proba


def make_hf_dataset(df):
    if Dataset is None:
        raise ImportError("datasets/transformers not installed.")

    dataset = Dataset.from_pandas(df[["text", "label_id"]], preserve_index=False)
    dataset = dataset.rename_column("label_id", "labels")
    return dataset


def tokenize_datasets(train_dataset, val_dataset, test_dataset, tokenizer, max_length=40):
    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True) if val_dataset is not None else None
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    return train_dataset, val_dataset, test_dataset


def compute_hf_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
    }


def train_bert_classifier(
    train_df,
    val_df,
    model_name="ProsusAI/finbert",
    output_dir="./bert_results",
    label2id=LABEL2ID,
    id2label=ID2LABEL,
    max_length=40,
    learning_rate=2e-5,
    train_batch_size=16,
    eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    seed=42,
):
    set_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_dataset = make_hf_dataset(train_df)
    val_dataset = make_hf_dataset(val_df)

    train_dataset, val_dataset, _ = tokenize_datasets(
        train_dataset,
        val_dataset,
        train_dataset,
        tokenizer,
        max_length=max_length,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="none",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_hf_metrics,
    )

    trainer.train()

    return trainer, tokenizer


def predict_bert(trainer, tokenizer, test_df, max_length=40):
    test_dataset = make_hf_dataset(test_df)
    _, _, test_dataset = tokenize_datasets(
        test_dataset,
        None,
        test_dataset,
        tokenizer,
        max_length=max_length,
    )

    outputs = trainer.predict(test_dataset)

    logits = outputs.predictions
    y_true = outputs.label_ids
    y_pred = np.argmax(logits, axis=1)

    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    proba = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return y_true, y_pred, proba