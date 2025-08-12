from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = Flask(__name__)

# Prediction function
def predict_diabetes(model, scaler, input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    prediction = model.predict(input_scaled)
    return 'Diabetic' if prediction[0] == 1 else 'Non-Diabetic'

# Load data, model, and scaler
with open('diabetes_data.pkl', 'rb') as f:
    data = pickle.load(f)
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    accuracy = None
    dataset_head = None
    feature_importance_plot = None
    glucose_plot = None
    bmi_plot = None

    if request.method == 'POST':
        # Get form data
        input_data = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['dpf']),
            float(request.form['age'])
        ]
        
        # Make prediction
        prediction = predict_diabetes(model, scaler, input_data)

    # Calculate model accuracy
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    X_scaled = scaler.transform(X)
    accuracy = model.score(X_scaled, y)

    # Prepare dataset overview
    dataset_head = data.head().to_html(classes='table table-striped')

    # Feature importance plot
    feature_importance = pd.DataFrame({
        'Feature': data.drop('Outcome', axis=1).columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig1 = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance')
    feature_importance_plot = fig1.to_html(full_html=False)

    # Data distribution plots
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Glucose Distribution', 'BMI Distribution'))
    fig.add_trace(go.Histogram(x=data['Glucose'], nbinsx=30, name='Glucose'), row=1, col=1)
    fig.add_trace(go.Histogram(x=data['BMI'], nbinsx=30, name='BMI'), row=1, col=2)
    fig.update_layout(showlegend=False)
    distribution_plot = fig.to_html(full_html=False)

    return render_template('index.html', 
                         prediction=prediction,
                         accuracy=accuracy,
                         dataset_head=dataset_head,
                         feature_importance_plot=feature_importance_plot,
                         distribution_plot=distribution_plot)

if __name__ == '__main__':
    app.run(debug=True)