from flask import Flask, request, render_template
import torch
import torch.nn as nn
import os
import jinja2


# Define the CarPriceModel class
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


# Load the trained model and scaler
model = torch.load('saved/car_price_model.pth', weights_only=False)
scaler = torch.load('saved/scaler.pth', weights_only=False)

template_dir = os.path.join(os.path.dirname(__file__), 'templates')
jinja_env = jinja2.Environment(loader = jinja2.FileSystemLoader(template_dir), autoescape = True)

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
    template = jinja_env.get_template('index.html')
    return template.render()


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
        template = app.jinja_env.get_template('index.html')
        return template.render(prediction=f'Predicted Price: ${predicted_price:.2f}')


if __name__ == "__main__":
    app.run(debug=True)
