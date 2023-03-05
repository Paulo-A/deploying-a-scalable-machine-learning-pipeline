# Put the code for your API here.
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union 

from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd

from ml.data import process_data
from ml.model import inference, load

def hyphenazier(string: str) -> str:
    return string.replace('_', '-')

class CensusData(BaseModel):
    age: int = Field(example = 40)
    workclass: str = Field(example = 'Private')
    fnlgt: int = Field(example = 121772)
    education: str = Field(example = 'Assoc-voc')
    education_num: int = Field(example = 11)
    marital_status: str = Field(example = 'Married-civ-spouse')
    occupation: str = Field(example = 'Craft-repair')
    relationship: str = Field(example = 'Husband')
    race: str = Field(example = 'Asian-Pac-Islander')
    sex: str = Field(example = 'Male')
    capital_gain: int = Field(example = 0)
    capital_loss: int = Field(example = 0)
    hours_per_week: int = Field(example = 40)
    native_country: str = Field(example = 'United-States')

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
