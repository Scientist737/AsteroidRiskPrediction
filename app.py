import requests
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# NASA API Key
API_URL = "https://api.nasa.gov/neo/rest/v1/feed"
API_KEY = "j4g15mddc0pTC7FHM8pK9wFg630t03FegPa6x4Kz"  # Replace with your NASA API key

app = Flask(__name__)

# Fetch Data from NASA API
def fetch_asteroid_data(start_date, end_date):
    params = {"start_date": start_date, "end_date": end_date, "api_key": API_KEY}
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Process Data for Machine Learning
def process_data(data):
    asteroids = []
    
    for date in data.get("near_earth_objects", {}):
        for asteroid in data["near_earth_objects"][date]:
            if "close_approach_data" in asteroid and asteroid["close_approach_data"]:
                try:
                    diameter = asteroid["estimated_diameter"]["meters"]["estimated_diameter_max"]
                    velocity = float(asteroid["close_approach_data"][0]["relative_velocity"]["kilometers_per_second"])
                    distance = float(asteroid["close_approach_data"][0]["miss_distance"]["kilometers"])
                    risk = 1 if asteroid.get("is_potentially_hazardous_asteroid", False) else 0

                    asteroids.append([diameter, velocity, distance, risk])
                except (KeyError, IndexError, ValueError) as e:
                    print(f"Skipping asteroid due to missing data: {e}")

    if not asteroids:
        print("No valid asteroid data found.")
        return None

    return pd.DataFrame(asteroids, columns=["diameter", "velocity", "distance", "risk"])

# Train Model
def train_model(df):
    if df is None or df.empty:
        print("No data available for training.")
        return

    X = df[["diameter", "velocity", "distance"]]
    y = df["risk"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.2f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved as model.pkl")

# Load Model & Predict
def predict_risk(diameter, velocity, distance):
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)
        
        input_data = np.array([[diameter, velocity, distance]])
        prediction = model.predict(input_data)
        return "Potentially Hazardous" if prediction[0] == 1 else "Not Hazardous"
    except FileNotFoundError:
        return "Model file not found. Train the model first."

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        diameter = float(data["diameter"])
        velocity = float(data["velocity"])
        distance = float(data["distance"])

        result = predict_risk(diameter, velocity, distance)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask App
if __name__ == "__main__":
    # Train model once before starting
    if not os.path.exists("model.pkl"):
        print("Training model as model.pkl was not found...")
        data = fetch_asteroid_data("2025-03-01", "2025-03-07")
        if data:
            df = process_data(data)
            train_model(df)
    
    app.run(debug=True)
