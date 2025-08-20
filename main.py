import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from flask import Flask, render_template, request
from PIL import Image
from io import BytesIO

# flask code
app = Flask(__name__, template_folder="app/templates")

# efficientnet used
model = models.efficientnet_b0(pretrained=False)

# updating for 11 classes available in the dataset
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, 11)

# loaded with the weights trained with
MODEL_PATH = os.path.join("models", "efficientnet_medicinal_leaves_model.pth")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Make sure it's in your repo!")

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# processing image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# classes trained with
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

def predict_image(file, threshold=0.6):
    # Read image directly from uploaded file in memory
    image = Image.open(BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)  # get probabilities
        confidence, predicted = probs.max(1)

    # if confidence is too low → return rejection message
    if confidence.item() < threshold:
        return "Please provide a proper medicinal leaf image."
    return class_labels[predicted.item()]

# routing part
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    uploaded_file = None
    disclaimer = (
        " Note: This model only predicts among 11 medicinal leaf classes. "
        "It cannot recognize other types of objects."
    )
    if request.method == "POST":
        file = request.files.get("image")
        if file:
            try:
                prediction = predict_image(file)
                uploaded_file = file.filename
            except Exception as e:
                return f"<h3>Prediction error:</h3><pre>{str(e)}</pre>", 500
    return render_template("index.html",
                           prediction=prediction,
                           uploaded_file=uploaded_file,
                           disclaimer=disclaimer)

# check code
@app.route("/ping")
def ping():
    return "pong"

# main part
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
