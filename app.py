import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template
import os
import sys

def identity(x):
    return x

sys.modules['__main__'].identity = identity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "xgb_model.pkl")
columns_path = os.path.join(BASE_DIR, "model_columns.pkl")


print("Starting Flask App...")
print("Project directory:", BASE_DIR)

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(columns_path, "rb") as f:
    columns = pickle.load(f)

print("Model loaded successfully")

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():

    try:

        data = request.form.to_dict()
        df = pd.DataFrame([data])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        df = df.fillna(0)
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        prediction = model.predict(df)
        price = round(prediction[0], 2)
        return render_template(
            "home.html",
            prediction_text=f"Estimated House Price: ${price}"
        )

    except Exception as e:

        print("Prediction error:", e)
        return render_template(
            "home.html",
            prediction_text="Prediction failed"
        )


# -------------------------------------------------
# API Prediction
# -------------------------------------------------
@app.route('/predict_api', methods=['POST'])
def predict_api():

    try:

        data = request.json
        df = pd.DataFrame([data])

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        df = df.fillna(0)
        df = pd.get_dummies(df)
        df = df.reindex(columns=columns, fill_value=0)
        prediction = model.predict(df)
        return jsonify({
            "prediction": prediction.tolist()
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 10000))

    app.run(host="0.0.0.0", port=port)