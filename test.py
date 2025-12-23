import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys

MODEL_PATH = 'face_model.pth'
LABELS_FILE = 'labels.txt'
IMAGE_PATH = '1.jpg'
CONFIDENCE_THRESHOLD = 70.0


def load_labels(label_path):
    if not os.path.exists(label_path):
        print(f"Error: File '{label_path}' not found. Please run main.py first.")
        sys.exit(1)

    with open(label_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def load_model(model_path, num_classes):
    print(f"Loading model from: {model_path}")

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found")
        sys.exit(1)

    try:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    except RuntimeError as e:
        print(f"Error: Model architecture mismatch. Details: {e}")
        sys.exit(1)

    model.eval()
    return model


def process_image(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found")
        sys.exit(1)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image)
        input_tensor = input_tensor.unsqueeze(0)
        return input_tensor
    except Exception as e:
        print(f"Error reading image: {e}")
        sys.exit(1)


def main():
    class_names = load_labels(LABELS_FILE)
    num_classes = len(class_names)
    print(f"Classes ({num_classes} people): {class_names}")

    model = load_model(MODEL_PATH, num_classes)

    input_tensor = process_image(IMAGE_PATH)

    print("-" * 30)
    print(f"Classifying image: {IMAGE_PATH} ...")

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_prob, top_idx = torch.max(probabilities, 0)

        confidence = top_prob.item() * 100

        if confidence > CONFIDENCE_THRESHOLD:
            predicted_class = class_names[top_idx.item()]
        else:
            predicted_class = "Unknown person"

    print(f"==> RESULT: {predicted_class}")
    print(f"==> CONFIDENCE: {confidence:.2f}%")
    print("-" * 30)

    print("Probability details:")
    for i, name in enumerate(class_names):
        prob = probabilities[i].item() * 100
        print(f" - {name}: {prob:.2f}%")


if __name__ == "__main__":
    main()