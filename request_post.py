import requests
import json
import pandas as pd

data_columns = ["age","workclass","fnlgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]

input_data = pd.DataFrame(
    [[40,"Private",121772,"Assoc-voc",11,"Married-civ-spouse","Craft-repair","Husband","Asian-Pac-Islander","Male",0,0,40,"?"]],
    columns=data_columns
)
response = requests.post('https://udacity-project3-qchz.onrender.com/predict', data=json.dumps(input_data.iloc[0,:].to_dict()))

print('Test case >50k')
print(response.status_code)
print(response.json())
print('--------------------')

input_data = pd.DataFrame(
    [[39,"State-gov",77516,"Bachelors",13,"Never-married","Adm-clerical","Not-in-family","White","Male",2174,0,40,"United-States"]],
    columns=data_columns
)
response = requests.post('https://udacity-project3-qchz.onrender.com/predict', data=json.dumps(input_data.iloc[0,:].to_dict()))

print('Test case <=50k')
print(response.status_code)
print(response.json())