from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

app = Flask(__name__)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes
NUM_CLASSES = 1000

# Class labels (HAM10000)
class_labels = {
    0: "Melanocytic nevi (nv)",
    1: "Melanoma (mel)",
    2: "Benign keratosis-like lesions (bkl)",
    3: "Basal cell carcinoma (bcc)",
    4: "Actinic keratoses (akiec)",
    5: "Vascular lesions (vasc)",
    6: "Dermatofibroma (df)"
}

# Load MobileNet architecture
model = models.mobilenet_v2(weights=None)

# Replace classifier for 7 classes
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

# Load trained weights
model.load_state_dict(torch.load("mobilenet_model.pth", map_location=device))

model.to(device)
model.eval()

# Image transforms (same as training)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# Prediction function
def predict_image(image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        _, pred = torch.max(outputs,1)

    return class_labels[pred.item()]


# Home route
@app.route('/')
def home():
    return "Skin Cancer Classifier API running"


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return jsonify({"error":"No file uploaded"}), 400

    file = request.files['file']

    try:

        image = Image.open(file).convert("RGB")

        prediction = predict_image(image)

        return jsonify({
            "prediction": prediction
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)