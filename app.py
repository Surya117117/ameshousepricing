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
    try:
        data = request.form.to_dict()
        df = pd.DataFrame([data])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.fillna(0)
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        prediction = xgbmodel.predict(df)
        output = round(prediction[0], 2)
        return render_template(
            "home.html",
            prediction_text=f"Estimated House Price: ${output}"
        )

    except Exception as e:
        print("Prediction error:", e)

        return render_template(
            "home.html",
            prediction_text="Prediction failed. Check server logs."
        )
@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        df = pd.DataFrame([data])
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        prediction = xgbmodel.predict(df)
        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )