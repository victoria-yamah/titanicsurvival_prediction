from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model & scaler
model = joblib.load("model/titanic_survival_model.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        pclass = int(request.form["Pclass"])
        sex = int(request.form["Sex"])
        age = float(request.form["Age"])
        fare = float(request.form["Fare"])
        embarked = request.form["Embarked"]

        # One-hot encoding (must match training!)
        embarked_q = 1 if embarked == "Q" else 0
        embarked_s = 1 if embarked == "S" else 0

        features = np.array([[pclass, sex, age, fare, embarked_q, embarked_s]])
        features = scaler.transform(features)

        prediction = model.predict(features)[0]
        result = "Survived" if prediction == 1 else "Did Not Survive"

        return render_template(
            "index.html",
            prediction_text=f"Prediction: {result}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"⚠️ Error: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)
