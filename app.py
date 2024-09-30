from flask import Flask, request, render_template
import torch
#import pandas as pd

# Load the trained model and scaler
model = torch.load('saved/car_price_model.pth')
scaler = torch.load('saved/scaler.pth')

app = Flask(__name__)


# Prediction function
def predict_pytorch(model, scaler, input_data):
    input_data_scaled = scaler.transform([input_data])
    input_tensor = torch.tensor(input_data_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        prediction = model(input_tensor)
    return prediction.item()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        year = int(request.form['year'])
        levy = float(request.form['levy'])
        manufacturer = request.form['manufacturer']
        model_name = request.form['model']
        prod_year = int(request.form['prod_year'])
        mileage = int(request.form['mileage'])
        cylinders = int(request.form['cylinders'])
        airbags = int(request.form['airbags'])

        input_data = [year, levy, manufacturer, model_name, prod_year, mileage, cylinders, airbags]
        predicted_price = predict_pytorch(model, scaler, input_data)

        return render_template('index.html', prediction=f'Predicted Price: ${predicted_price:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
