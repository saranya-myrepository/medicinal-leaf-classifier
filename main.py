import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from flask import Flask, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

# ---------------------------
# 1. Flask Setup
# ---------------------------
app = Flask(__name__, template_folder="app/templates")  # explicitly point to templates

# ---------------------------
# 2. Model Setup
# ---------------------------
# Load base EfficientNet
model = models.efficientnet_b0(pretrained=False)

# Update classifier to match 11 classes
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 11)

# Load trained weights
MODEL_PATH = os.path.join("models", "efficientnet_medicinal_leaves_model.pth")
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# ---------------------------
# 3. Image Preprocessing
# ---------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Medicinal plant classes (must match training order)
class_labels = [
    "Balloon Vine",
    "Bringaraja",
    "Caricature",
    "Catharanthus",
    "Drumstick",
    "Eucalyptus",
    "Ganigale",
    "Henna",
    "Hibiscus",
    "Rose",
    "Spinach1"
]

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    return class_labels[predicted.item()]

# ---------------------------
# 4. Flask Routes
# ---------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_file = None
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                upload_dir = "uploads"
                os.makedirs(upload_dir, exist_ok=True)
                filename = secure_filename(file.filename)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)

                prediction = predict_image(filepath)
                uploaded_file = filename
            except Exception as e:
                # Return detailed error message in browser for debugging
                return f"<h2>Error during prediction:</h2><pre>{str(e)}</pre>", 500
    return render_template("index.html", prediction=prediction, uploaded_file=uploaded_file)

# Health check route for debugging
@app.route("/ping")
def ping():
    return "pong"

# ---------------------------
# 5. Main Entry
# ---------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
