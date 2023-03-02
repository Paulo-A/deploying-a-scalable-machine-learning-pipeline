import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference

data = pd.read_csv('./data/census.csv')

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

def test_train_model():
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)

def inference():
    model = train_model(X_train, y_train)

    preds = inference(model, y_test)

    assert isinstance(preds, np.array)


def test_compute_model_metrics():
    model = train_model(X_train, y_train)
    preds = inference(model, y_test)

    precision, recall, fbeta = compute_model_metrics(y_train, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)