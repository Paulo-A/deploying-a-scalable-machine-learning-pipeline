# Script to train machine learning model.

from sklearn.model_selection import train_test_split
import pandas as pd

from ml.data import process_data
from ml.model import train_model, save_model

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

model = train_model(X_train, y_train)
save_model(model, '../model/random_forest.pkl')
