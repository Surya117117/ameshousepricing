import pickle
from flask import Flask, request, jsonify, url_for, render_template

import numpy as np
import pandas as pd

app = Flask(__name__)
# Load the model
xgbmodel = pickle.load(open('xgb_model.pkl', 'rb'))
columns = pickle.load(open('model_columns.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])

def predict_api():
    data = request.json['data']
    print(data)
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)
    prediction = xgbmodel.predict(df)
    return jsonify({"prediction": prediction.tolist()})
    
if __name__ == "__main__":
    app.run(debug=True)
