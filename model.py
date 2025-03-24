import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Define API details
API_URL = "https://api.nasa.gov/neo/rest/v1/feed"
API_KEY = "j4g15mddc0pTC7FHM8pK9wFg630t03FegPa6x4Kz"  # Replace with a valid NASA API key

# Fetch data from API
def fetch_asteroid_data(start_date, end_date):
    params = {"start_date": start_date, "end_date": end_date, "api_key": API_KEY}
    
    try:
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Raise exception for HTTP errors (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

# Process data
def process_data(data):
    asteroids = []
    
    for date in data.get("near_earth_objects", {}):
        for asteroid in data["near_earth_objects"][date]:
            # Ensure the asteroid has necessary data
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

    df = pd.DataFrame(asteroids, columns=["diameter", "velocity", "distance", "risk"])
    return df

# Train and Save ML Model
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
    print(df["risk"].value_counts())

  
# Load Model and Make Predictions
import pickle
import numpy as np
import pandas as pd

def predict_risk(diameter, velocity, distance):
    try:
        with open("model.pkl", "rb") as f:
            model = pickle.load(f)

        # Convert input to a DataFrame with correct column names
        input_data = pd.DataFrame([[diameter, velocity, distance]], columns=["diameter", "velocity", "distance"])

        prediction = model.predict(input_data)
        proba = model.predict_proba(input_data)

        print(f"Confidence: {proba}")
        return "Potentially Hazardous" if prediction[0] == 1 else "Not Hazardous"

    except FileNotFoundError:
        print("Model file not found. Train the model first.")
        return "Model file not found. Train the model first."

# Test cases
test_cases = [
    [500, 25, 2000000],
    [1200, 35, 500000],
    [150, 15, 5000000],
    [900, 40, 1000000],
    [2000, 28, 750000],
]

for case in test_cases:
    result = predict_risk(*case)
    print(f"Input: {case}, Prediction: {result}")



# Run the pipeline
if __name__ == "__main__":
    data = fetch_asteroid_data("2025-03-01", "2025-03-07")
    if data:
        df = process_data(data)
        train_model(df)

        # Example Prediction
        result = predict_risk(500, 25, 300000)
        print(f"Prediction Result: {result}")
