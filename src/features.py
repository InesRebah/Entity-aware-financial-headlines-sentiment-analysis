import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def target_position(text):
    words = re.sub(r"[,:;]", "", str(text)).split()
    
    if "TARGET" not in words or len(words) == 0:
        return None
    
    return words.index("TARGET") / len(words)

def other_numbers(text):    
    words = re.sub(r"[,:;]", "", str(text)).split()
    
    if "OTHER" not in words:
        return 0
    
    return words.count('OTHER')

def get_custom_stopwords(keep_words=None):
    """
    Build English stopwords while keeping task-specific tokens such as OTHER.
    """
    if keep_words is None:
        keep_words = {"other", "target"}

    return list(set(ENGLISH_STOP_WORDS) - set(keep_words))


def build_tfidf(texts, max_features=10000, stop_words=None):
    """
    Fit TF-IDF vectorizer and return matrix, vectorizer, feature names.
    """
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words=stop_words
                )

    X = tfidf.fit_transform(texts)
    feature_names = tfidf.get_feature_names_out()

    return X, tfidf, feature_names


def get_global_top_tfidf_words(X, feature_names, top_n=20):
    """
    Compute globally highest average TF-IDF words.
    """
    scores = np.asarray(X.mean(axis=0)).ravel()
    top_idx = scores.argsort()[-top_n:][::-1]

    return pd.DataFrame({
        "word": feature_names[top_idx],
        "score": scores[top_idx]
    })


def top_words_by_class(X, labels, feature_names, label, top_n=10):
    """
    Compute top average TF-IDF words for one sentiment class.
    """
    labels = np.asarray(labels)
    mask = labels == label

    X_label = X[mask]
    scores = np.asarray(X_label.mean(axis=0)).ravel()
    top_idx = scores.argsort()[-top_n:][::-1]

    return pd.DataFrame({
        "word": feature_names[top_idx],
        "score": scores[top_idx],
        "label": label
    })


def get_top_words_all_classes(X, labels, feature_names, top_n=10):
    """
    Compute top TF-IDF words for all sentiment classes.
    """
    classes = ["negative", "neutral", "positive"]

    dfs = [
        top_words_by_class(X, labels, feature_names, label=c, top_n=top_n)
        for c in classes
    ]

    return pd.concat(dfs, ignore_index=True)


def fit_lsa(X, n_components=50, random_state=42):
    """
    Fit LSA using TruncatedSVD.
    """
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=random_state
    )

    X_lsa = svd.fit_transform(X)

    return X_lsa, svd


def build_topic_dataframe(X_lsa, labels):
    """
    Create dataframe of LSA topic values with labels.
    """
    df_topics = pd.DataFrame(
        X_lsa,
        columns=[f"topic_{i}" for i in range(X_lsa.shape[1])]
    )
    df_topics["label"] = np.asarray(labels)

    return df_topics


def standardize_topic_means(df_mean):
    """
    Standardize topic means column-wise for heatmap visualization.
    """
    scaler = StandardScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_mean),
        index=df_mean.index,
        columns=df_mean.columns
    )

    return df_scaled


def get_topic_terms(svd, feature_names, topic_idx, top_n=10):
    """
    Extract positive and negative sides of one LSA topic.
    """
    component = svd.components_[topic_idx]

    top_pos_idx = component.argsort()[-top_n:][::-1]
    top_neg_idx = component.argsort()[:top_n]

    positive_side = pd.DataFrame({
        "word": feature_names[top_pos_idx],
        "weight": component[top_pos_idx],
        "side": "positive"
    })

    negative_side = pd.DataFrame({
        "word": feature_names[top_neg_idx],
        "weight": component[top_neg_idx],
        "side": "negative"
    })

    return positive_side, negative_side


def print_topics(svd, feature_names, n_topics=10, top_n=10, both_sides=False):
    """
    Print top words for several LSA topics.
    """
    for i, component in enumerate(svd.components_[:n_topics]):
        print(f"\nTopic {i}")

        if both_sides:
            pos, neg = get_topic_terms(svd, feature_names, i, top_n=top_n)

            print("Positive side:")
            print(pos["word"].tolist())

            print("Negative side:")
            print(neg["word"].tolist())

        else:
            top_idx = component.argsort()[-top_n:][::-1]
            print(feature_names[top_idx].tolist())