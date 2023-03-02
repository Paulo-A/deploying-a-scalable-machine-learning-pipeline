from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from ml.data import process_data

import pickle


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    random_forest_classifier = RandomForestClassifier()
    random_forest_classifier.fit(X_train, y_train)

    return random_forest_classifier

def save_model(model, filepath):
    """ Save model as pickle in filepath

    Inputs
    ------
    model : ???
        Trained machine learning model.
    filepath : str
        Path for the file.
    Returns
    -------
    """
    with open(filepath, 'wb') as model_file:
        pickle.dump(model, model_file)

def compute_model_performance_on_slices(data, features, cat_features, model):
    """
    Compute models performance with precision, recall, and F1 by slicing data.

    Inputs
    ------
    data : pd.DataFrame
        Unprocessed data to compute metrics
    features : list[str]
        Name of the features to slice.
    categorical_features : list[str]
        List containing the names of the categorical features (default=[])
    model : ???
        Trained machine learning model.
    Returns
    -------
    model_performance_df : pd.DataFrame
        DataFrame with the evaluated performance.
    """
    model_performance_df = pd.DataFrame(columns=['constant_column','precision', 'recall', 'fbeta'])

    all_performance = ''
    for feature in features:
        data_to_test = data[data["class"] == feature]

        X, y, encoder, lb = process_data(
            data_to_test, categorical_features=cat_features, label="salary", training=True
        )

        preds = inference(model, X)

        precision, recall, fbeta = compute_model_metrics(X, preds)

        run_performance = f'''
        Feature {feature}:
        \tprecision: {precision}'
        \trecall: {recall}
        \fbeta: {fbeta}

        '''
        print(run_performance)

        all_performance += run_performance
        model_performance_df.append([feature, precision, recall, fbeta])

    with open('slice_output.txt', 'wb') as performance_file:
        performance_file.write(all_performance)

    return model_performance_df

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
