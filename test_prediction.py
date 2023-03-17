import requests
import json
import numpy as np

# define the input data as a numpy array
input_data = np.array([[3, 45.85, 142.8, 0]])

# send a POST request to the Flask app with the input data
response = requests.post('http://127.0.0.1:5000/predict', json=input_data.tolist())

# get the predicted churn status from the response
churn_prediction = response.json()['churn_prediction']

# print the predicted churn status
if churn_prediction:
    print("The customer is predicted to churn.")
else:
    print("The customer is predicted to stay with the company.")
