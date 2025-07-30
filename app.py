import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Load model and scaler
logistic_model = pickle.load(open('models/logistic.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        # Fetch form inputs
        pclass = float(request.form.get('pclass', '0'))
        age = float(request.form.get('age', '0'))
        sibsp = float(request.form.get('sibsp', '0'))
        parch = float(request.form.get('parch', '0'))
        fare = float(request.form.get('fare', '0'))

        # Encode categorical variables
        sex_str = request.form.get('sex', 'male').lower()
        embarked_str = request.form.get('embarked', 's').lower()
        who_str = request.form.get('who', 'man').lower()
        alone_str = request.form.get('alone', 'false').lower()

        # Manual encoding to numeric
        sex = 1.0 if sex_str == 'male' else 0.0
        embarked = {'s': 0.0, 'c': 1.0, 'q': 2.0}.get(embarked_str, 0.0)
        who = {'man': 0.0, 'woman': 1.0, 'child': 2.0}.get(who_str, 0.0)
        alone = 1.0 if alone_str == 'true' else 0.0

        # Combine into feature vector
        input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked, who, alone]])

        # Scale and predict
        new_data_scaled = standard_scaler.transform(input_data)
        result = logistic_model.predict(new_data_scaled)

        return render_template('home.html', result=result[0])

    else:
        return render_template('home.html')


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
