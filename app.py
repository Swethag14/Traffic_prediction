from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the models
with open('lr_model.pkl', 'rb') as f:
    lr_model = pickle.load(f)

with open('rf_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('dbscan_model.pkl', 'rb') as f:
    dbscan_model = pickle.load(f)

# Function to map predictions to traffic situations
def get_traffic_situation(pred):
    mapping = {0: "Low", 1: "Medium", 2: "High"}
    return mapping.get(pred, "Unknown")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    # Ensure keys in 'data' match the trained model's feature names
    data = {
        'Hour': data['Hour'],
        'Date': data['Date'],
        'Day of the week': data['Day_of_the_week'],  # Consistency in naming
        'CarCount': data['CarCount'],
        'BikeCount': data['BikeCount'],
        'BusCount': data['BusCount'],
        'TruckCount': data['TruckCount']
    }
    input_data = pd.DataFrame([data])
    input_data = input_data.astype(float)

    lr_prediction = int(lr_model.predict(input_data)[0])
    rf_prediction = int(rf_model.predict(input_data)[0])
    kmeans_prediction = int(kmeans_model.predict(input_data)[0])
    dbscan_prediction = int(dbscan_model.fit_predict(input_data)[0])

    # Get the traffic situation from one of the models, e.g., Random Forest
    traffic_situation = get_traffic_situation(rf_prediction)

    return jsonify({
        'Traffic Situation': traffic_situation
    })

if __name__ == '__main__':
    app.run(debug=True)
