# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- Model date: 2023-03-05
- Model version: 1.0.0
- Model type: sklearn.ensemble.RandomForestClassifier

## Intended Use
- Solution for the third project of Udacity's Deploying a Scalable ML Pipeline in Production Nanodegree

## Training Data
- Census Income Data Set provided by UCI at https://archive.ics.uci.edu/ml/datasets/census+income

## Evaluation Data
- Dataset split into 20% for testing and 80% for training
- Data preprocessed with:
  - OneHotEncoder(sparse=False, handle_unknown="ignore") for categorical data
  - Returns: 0 if salary >50k and 1 if salary <=50k

## Metrics
- precision: 0.755
- recall: 0.633
- fbeta: 0.689

## Ethical Considerations
- This solution is not to be used on other Udacity's projects.

## Caveats and Recommendations
