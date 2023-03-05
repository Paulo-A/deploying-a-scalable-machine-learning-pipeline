# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd

from ml.data import process_data
from ml.model import train_model, save_model, inference, compute_model_metrics

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

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

model = train_model(X_train, y_train)

preds = inference(model, X_test)

precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(precision, recall, fbeta)
save_model(model, './model/random_forest.pkl')
save_model(encoder, './model/encoder.pkl')
save_model(lb, './model/label_binarizer.pkl')
