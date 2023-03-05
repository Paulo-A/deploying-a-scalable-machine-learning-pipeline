from fastapi.testclient import TestClient
import pandas as pd

# Import our app from main.py.
from main import app

# Instantiate the testing client with our app.
client = TestClient(app)

data_columns = ["age","workclass","fnlgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]
# Write tests using the same syntax as with the requests module.
def test_get_greet_user():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Greetings": "This app is a solution for Udacity's Deploying a Scalable ML Pipeline in Production Nanodegree"}

def test_post_model_predict_higher():
    input_data = pd.DataFrame(
        [[40,"Private",121772,"Assoc-voc",11,"Married-civ-spouse","Craft-repair","Husband","Asian-Pac-Islander","Male",0,0,40,"?"]],
        columns=data_columns
    )
    r = client.post("/predict", json=input_data.iloc[0,:].to_dict())

    assert r.status_code == 200
    assert r.json() == {"inference_result": {'0': ">50k"}}

def test_post_model_predict_lower():
    input_data = pd.DataFrame(
        [[39,"State-gov",77516,"Bachelors",13,"Never-married","Adm-clerical","Not-in-family","White","Male",2174,0,40,"United-States"]],
        columns=data_columns
    )
    r = client.post("/predict", json=input_data.iloc[0,:].to_dict())

    assert r.status_code == 200
    assert r.json() == {"inference_result": {'0': "<=50k"}}