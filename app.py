import joblib
import numpy as np
from flask import Flask, jsonify, request

# Load the trained model and scaler
model = joblib.load('decision_tree_model.joblib')
scaler = joblib.load('feature_scaler.joblib')

# Define the Flask application
app = Flask(__name__)

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Parse the input data from the request
    input_data = request.json
    X = np.array(input_data).reshape(1, -1)

    # Scale the input features
    X_scaled = scaler.transform(X)

    # Make the prediction using the trained model
    y_pred = model.predict(X_scaled)

    # Convert the prediction to a JSON response
    response = {'churn_prediction': bool(y_pred[0])}
    return jsonify(response)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)