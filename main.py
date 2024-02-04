"""
A program that predicts the price of used cars based on year, make, model, mileage, etc.

author: Quang Huynh
started: 02/04/2024
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

def load_data(fileName):
    try:
        data = pd.read_csv(fileName)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    # Check if 'make' column exists
    if 'make' in df.columns:
        # Encode categorical variables
        label_encoder = LabelEncoder()
        df['make'] = label_encoder.fit_transform(df['make'])
        df['model'] = label_encoder.fit_transform(df['model'])
        return df
    else:
        print("Error: 'make' column not found.")
        return None

def train_model(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

def predict_price(model, input_data):
    label_encoder = LabelEncoder()
    input_data['make'] = label_encoder.transform([input_data['make']])
    input_data['model'] = label_encoder.transform([input_data['model']])

    prediction = model.predict([input_data.values])
    return prediction[0]

if __name__ == "__main__":
    # Load dataset
    df = load_data('usedcars.csv')  # Replace with your dataset path

    if df is not None:
        # Preprocess data
        df = preprocess_data(df)

        if df is not None:
            # Split into features (X) and target variable (y)
            X = df.drop('price', axis=1)
            y = df['price']

            # Train the model
            model = train_model(X, y)

            # Example usage for prediction
            new_car_data = {'year': 2018, 'make': 'Toyota', 'model': 'Camry', 'mileage': 50000, '...': '...'}
            predicted_price = predict_price(model, new_car_data)

            print(f'Predicted Price: ${predicted_price:.2f}')
