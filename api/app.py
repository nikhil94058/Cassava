import os
import torch
import torchvision.transforms as transforms
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from PIL import Image
from torchvision.models import shufflenet_v2_x1_0
import torch.nn as nn

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS globally

# Define device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the model architecture
class SSDClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(SSDClassifier, self).__init__()
        in_features = base_model.fc.in_features  # Get feature count before replacing FC layer
        base_model.fc = nn.Identity()  # Remove final FC layer
        self.feature_extractor = base_model
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        return self.classifier(features)

# Load trained model
num_classes = 5  # Change based on your dataset
base_model = shufflenet_v2_x1_0(pretrained=False)  # Ensure the same architecture
model = SSDClassifier(base_model, num_classes)

# Load model weights
model_path = "cassava_model_trained_with_gpu.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    raise FileNotFoundError(f"Model file '{model_path}' not found!")

model.to(device)
model.eval()

# Define class labels
class_labels = ["Cassava Bacterial Blight", "Cassava Brown Streak Disease", 
                "Cassava Green Mottle", "Cassava Mosaic Disease", "Healthy"]

@app.route('/')
def home():
    return "Cassava Leaf Disease Classification API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']

    try:
        # Open and preprocess the image
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            predicted_class = class_labels[predicted.item()]
            confidence = torch.softmax(outputs, dim=1)[0][predicted.item()].item() * 100

        return jsonify({"prediction": predicted_class, "confidence": round(confidence, 2)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
