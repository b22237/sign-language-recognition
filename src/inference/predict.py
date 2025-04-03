import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import models, transforms
from transformers import BertModel, BertConfig
import pickle
from PIL import Image
import argparse
# Load gesture label mapping
with open("data/labels/word_to_label.pkl", "rb") as f:
    word_to_label = pickle.load(f)
label_to_word = {v: k for k, v in word_to_label.items()}  # Reverse mapping
parser = argparse.ArgumentParser(description='Process segmentation masks and images.')
parser.add_argument('-i', '--input', required=True, help='Output directory for images')
args = parser.parse_args()

# Load trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GestureTransformer(nn.Module):
    def __init__(self, num_classes, feature_dim=2048, seq_len=50, hidden_dim=768):
        super(GestureTransformer, self).__init__()

        self.input_proj = nn.Linear(feature_dim, hidden_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x.permute(1, 0, 2))  # (seq_len, batch, hidden_dim)
        x = x.mean(dim=0)  # Pooling over sequence
        return self.fc(x)


# Load model
num_classes = len(word_to_label)
model = GestureTransformer(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("/home/soham/garments/preet/preet1/sign-language-detection/models/gesture_transformer.pth"))
model.eval()

# Load ResNet50 for feature extraction
resnet = models.resnet50(pretrained=True)
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove final FC layer
resnet.eval().to(device)

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_features(image_path):
    """Preprocess image and extract features using ResNet."""
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet(image)

    return features.view(-1, 2048).cpu().numpy()  # Flatten to (2048,)

def predict_gesture(features):
    """Predict gesture using the trained transformer model."""
    max_seq_len = 50
    feature_dim = 2048

    # Prepare input tensor (pad or truncate to 50 frames)
    padded_features = np.zeros((max_seq_len, feature_dim))
    length = min(len(features), max_seq_len)
    padded_features[:length] = features[:length]

    # Convert to tensor and pass through model
    input_tensor = torch.tensor(padded_features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        predicted_idx = torch.argmax(logits, dim=1).item()

    return label_to_word[predicted_idx]

# Directory containing hand-segmented images
input_folder = os.path.join(args.input)

# Collect features for the entire sequence (video)
features_list = []

for img_file in sorted(os.listdir(input_folder)):  # Ensure correct sequence order
    if img_file.endswith('.jpg'):
        img_path = os.path.join(input_folder, img_file)
        print(img_path)
        features = extract_features(img_path)
        features_list.append(features)

# Convert list of features to a numpy array (sequence_length, feature_dim)
features_seq = np.vstack(features_list) if features_list else np.zeros((1, 2048))  # Handle empty folder case

# Predict gesture for the entire sequence
predicted_gesture = predict_gesture(features_seq)

print(f"Predicted Gesture for Video: {predicted_gesture}")

