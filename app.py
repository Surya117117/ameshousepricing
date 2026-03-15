import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

def identity(x):
    return x

app = Flask(__name__)

# Load trained model and training columns
xgbmodel = pickle.load(open("xgb_model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))



# Home page
@app.route('/')
def home():
    return render_template("home.html")


# Prediction from HTML form
@app.route('/predict', methods=['POST'])
def predict():

    # Get data from HTML form
    data = request.form.to_dict()

    # Convert to dataframe
    df = pd.DataFrame([data])

    # Convert numeric strings to numbers
    df = df.apply(pd.to_numeric, errors='ignore')

    # Apply one-hot encoding
    df = pd.get_dummies(df)

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Make prediction
    prediction = xgbmodel.predict(df)

    return render_template(
        "home.html",
        prediction_text=f"The House price prediction is {prediction[0]}"
    )


# Prediction API (for Postman or external requests)
@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']

    # Convert to dataframe
    df = pd.DataFrame([data])

    # Apply encoding
    df = pd.get_dummies(df)

    # Align columns
    df = df.reindex(columns=columns, fill_value=0)

    prediction = xgbmodel.predict(df)

    return jsonify({
        "prediction": prediction.tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)