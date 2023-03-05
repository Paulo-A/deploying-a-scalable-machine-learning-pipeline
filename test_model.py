import pytest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, load, compute_model_performance_on_slices

data = pd.read_csv('./data/census.csv')
model = load('./model/random_forest.pkl')

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

features = train.columns

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

def test_train_model():
    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)

def test_inference():
    preds = inference(model, X_test)

    assert len(preds) == len(y_test)
    assert isinstance(preds, np.ndarray)

def test_compute_model_metrics():
    preds = inference(model, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)

def test_compute_model_performance_on_slices():
    performance_df = compute_model_performance_on_slices(data=data, label="salary", features=cat_features, cat_features=cat_features, model=model, encoder=encoder, lb=lb)
    assert isinstance(performance_df, pd.DataFrame)
