from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torchvision.transforms as transforms
import io
import os
from torchvision import models
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define model architecture
class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.network = models.resnet50(pretrained=False)  # Pretrained set to False
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.network(x)

# Load the model
model_path = 'model_weights.pth'  # Update with the path to your pickle file
num_classes = 6  # Adjust according to your dataset
model = ResNet50(num_classes=num_classes)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # Read image
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read())).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        label = predicted.item()

    # Map label to expression
    expression_mapping = {
        0: "Anger",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Pain",
        5: "Sad"
    }
    prediction = expression_mapping.get(label, "Unknown")

    return jsonify({"expression": prediction})

if __name__ == '__main__':
    app.run(debug=True)
