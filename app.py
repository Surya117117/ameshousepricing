import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import os

def identity(x):
    return x

app = Flask(__name__)

try:
    xgbmodel = pickle.load(open("xgb_model.pkl", "rb"))
    columns = pickle.load(open("model_columns.pkl", "rb"))
    print("Model loaded successfully")
except Exception as e:
    print("Model loading failed:", e)
    raise e


@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict', methods=['POST'])
def predict():

    data = request.form.to_dict()
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = xgbmodel.predict(df)
    price = np.exp(prediction[0])

    return render_template(
        "home.html",
        prediction_text=f"Predicted House Price: ${round(price,2)}"
    )

@app.route('/predict_api', methods=['POST'])
def predict_api():

    data = request.json['data']
    df = pd.DataFrame([data])
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.fillna(0)
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    prediction = xgbmodel.predict(df)
    price = np.exp(prediction[0])
    return jsonify({
        "prediction": float(price)
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)