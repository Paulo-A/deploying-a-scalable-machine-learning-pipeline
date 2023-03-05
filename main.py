# Put the code for your API here.
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd

from ml.data import process_data
from ml.model import inference, load

def hyphenazier(string: str) -> str:
    return string.replace('_', '-')

class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = hyphenazier
        allow_population_by_field_name = True

app = FastAPI()
model = load('./model/random_forest.pkl')
encoder = load('./model/encoder.pkl')
lb = load('./model/label_binarizer.pkl')

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

@app.get("/")
async def greet_user():
    return {"Greetings": "This app is a solution for Udacity's Deploying a Scalable ML Pipeline in Production Nanodegree"}

@app.post("/predict")
async def model_predict(endpoint_input: CensusData):
    endpoint_input_dict = endpoint_input.dict(by_alias=True)
    model_input = pd.DataFrame([endpoint_input_dict])

    processed_model_input, _, _, _ = process_data(
        model_input, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )

    inference_result_list = list(inference(model, processed_model_input))

    inference_result = {}
    for i in range(len(inference_result_list)):
        if inference_result_list[i] == 0:
            inference_result[i] = '<=50k'
        else:
            inference_result[i] = '>50k'

    return {"inference_result": inference_result}
