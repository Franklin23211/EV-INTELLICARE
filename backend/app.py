from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoder
model = joblib.load('../models/vehicle_health_model.pkl')
label_encoder = joblib.load('../models/label_encoder.pkl')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values
    battery_health = float(request.form['battery_health'])
    motor_temp = float(request.form['motor_temp'])
    brake_wear = float(request.form['brake_wear'])
    suspension_stress = float(request.form['suspension_stress'])
    tire_pressure = float(request.form['tire_pressure'])
    past_failures = int(request.form['past_failures'])

    features = [battery_health, motor_temp, brake_wear, suspension_stress, tire_pressure, past_failures]
    prediction = model.predict([features])[0]
    label = label_encoder.inverse_transform([prediction])[0]

    # Precaution logic for Warning or Critical
    precaution = ""
    if label in ["Warning", "Critical"]:
        issues = []
        if battery_health < 50:
            issues.append("🔋 Battery health is low. Consider checking or replacing the battery.")
        if motor_temp > 90:
            issues.append("🌡️ Motor is overheating. Check cooling system or riding conditions.")
        if brake_wear > 70:
            issues.append("🛑 Brakes are heavily worn. Immediate servicing recommended.")
        if suspension_stress > 60:
            issues.append("🔧 Suspension is under high stress. May affect ride comfort and control.")
        if tire_pressure < 28:
            issues.append("⚠️ Tire pressure is low. Inflate tires to avoid handling issues.")

        precaution = "<br>".join(issues)

    return render_template("result.html", status=label, precaution=precaution)

if __name__ == "__main__":
    app.run(debug=True)
