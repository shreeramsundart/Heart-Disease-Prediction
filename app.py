from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

app.secret_key = 'heart_disease_prediction'  # Required for sessions

# Load ML model and scaler
scaler = joblib.load("model/scaler_heart.pkl")
model = joblib.load("model/heart_disease_prediction.pkl")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get user inputs from the form
            age = int(request.form["age"])
            sex = 1 if request.form["sex"] == "Male" else 0
            cp = int(request.form["cp"])
            trestbps = int(request.form["trestbps"])
            chol = int(request.form["chol"])
            fbs = 1 if request.form["fbs"] == "Yes" else 0
            restecg = int(request.form["restecg"])
            thalach = int(request.form["thalach"])
            exang = 1 if request.form["exang"] == "Yes" else 0
            oldpeak = float(request.form["oldpeak"])
            slope = int(request.form["slope"])
            ca = int(request.form["ca"])
            thal = int(request.form["thal"])

            # Prepare data for the model
            values = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            scaled_values = scaler.transform(values)

            # Make prediction
            prediction = model.predict(scaled_values)[0]

            # Determine heart disease risk
            if prediction == 0:
                risk = "Low Risk"
            else:
                risk = "High Risk"

            return render_template("index.html", risk=risk, prediction=prediction)

        except Exception as e:
            return render_template("index.html", error=f"Error: {e}")

    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)