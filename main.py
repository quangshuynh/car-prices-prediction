"""
A program that predicts the price of used cars based on year, make, model, mileage, etc.

author: Quang Huynh
started: 02/04/2024
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
import torch.nn as nn


def preprocess_data(df):
    # Handle missing values in 'Levy' by replacing '-' with 0 and converting the column to numeric
    df['Levy'] = pd.to_numeric(df['Levy'].replace('-', 0))

    # Clean 'Engine volume' by extracting the numeric part and converting it to float
    df['Engine volume'] = df['Engine volume'].str.extract(r'(\d+\.\d+)').astype(float)

    # Clean 'Mileage' by removing the ' km' suffix and converting it to numeric
    df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(float)

    # Clean 'Doors' by replacing non-numeric values and converting to numeric
    df['Doors'] = pd.to_numeric(df['Doors'], errors='coerce').fillna(4)  # Default to 4 doors if not specified

    # Encode categorical variables using LabelEncoder
    label_encoder = LabelEncoder()
    df['Manufacturer'] = label_encoder.fit_transform(df['Manufacturer'])
    df['Model'] = label_encoder.fit_transform(df['Model'])
    df['Category'] = label_encoder.fit_transform(df['Category'])
    df['Leather interior'] = label_encoder.fit_transform(df['Leather interior'])
    df['Fuel type'] = label_encoder.fit_transform(df['Fuel type'])
    df['Gear box type'] = label_encoder.fit_transform(df['Gear box type'])
    df['Drive wheels'] = label_encoder.fit_transform(df['Drive wheels'])
    df['Wheel'] = label_encoder.fit_transform(df['Wheel'])
    df['Color'] = label_encoder.fit_transform(df['Color'])

    # Drop unnecessary columns
    df = df.drop(['ID'], axis=1)

    # Separate features (X) and target (y)
    X = df.drop('Price', axis=1)
    y = df['Price']

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


# Split the dataset into training and testing sets
def split_data(df):
    X, y, scaler = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, scaler


# Define a simple neural network model using PyTorch
class CarPriceModel(nn.Module):
    def __init__(self, input_size):
        super(CarPriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Train the model
def train_pytorch_model(X_train, y_train, input_size, epochs=50, lr=0.001):
    model = CarPriceModel(input_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    return model


# Main function to train and save the model
def main():
    # Load dataset
    df = pd.read_csv('csv/train.csv')  # Use your path here

    # Preprocess and split the data
    X_train, X_test, y_train, y_test, scaler = split_data(df)

    # Train the model
    input_size = X_train.shape[1]
    model = train_pytorch_model(X_train, y_train, input_size)

    # Save the model and scaler
    torch.save(model, 'saved/car_price_model.pth')
    torch.save(scaler, 'saved/scaler.pth')

    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    main()
