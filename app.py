import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

def identity(x):
    return x

app = Flask(__name__)

xgbmodel = pickle.load(open("xgb_model.pkl", "rb"))
columns = pickle.load(open("model_columns.pkl", "rb"))


@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():

    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric, errors='ignore')
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = xgbmodel.predict(df)
    return render_template(
        "home.html",
        prediction_text=f"The House price prediction is {prediction[0]}"
    )

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = xgbmodel.predict(df)
    return jsonify({
        "prediction": prediction.tolist()
    })

if __name__ == "__main__":
    app.run(debug=True)