import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import time
import argparse
from typing import List, Tuple, Optional

class FaceRecognizerCUDA:
    def __init__(self, model_path: str, labels_path: str, device_name: str = "cuda"):
        if not torch.cuda.is_available() and device_name == "cuda":
            print("Warning: CUDA not available, switching to CPU.")
            device_name = "cpu"
        
        self.device = torch.device(device_name)
        self.labels = self._load_labels(labels_path)
        self.model = self._load_model(model_path, len(self.labels))
        
        # Standard ImageNet normalization
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Warm-up GPU
        self._warmup()

    def _load_labels(self, path: str) -> List[str]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Labels file not found: {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_model(self, path: str, num_classes: int) -> nn.Module:
        print(f"Loading model to {self.device}...")
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            model.load_state_dict(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load weights: {e}")

        model.to(self.device)
        model.eval()
        return model

    def _warmup(self):
        dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
        with torch.no_grad():
            self.model(dummy_input)

    def predict(self, image_path: str) -> Tuple[str, float, float, List[float]]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as e:
            raise ValueError(f"Error processing image: {e}")

        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            # Handle potential NaNs
            if torch.isnan(outputs).any():
                outputs = torch.nan_to_num(outputs, nan=0.0)

            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)
            
        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        predicted_label = self.labels[idx.item()]
        confidence = conf.item() * 100
        all_probs = [p.item() * 100 for p in probs]

        return predicted_label, confidence, inference_time_ms, all_probs

def main():
    parser = argparse.ArgumentParser(description="PyTorch CUDA Face Recognition Inference")
    parser.add_argument("--image", type=str, default="2.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="face_model.pth", help="Path to .pth model")
    parser.add_argument("--labels", type=str, default="labels.txt", help="Path to labels file")
    parser.add_argument("--threshold", type=float, default=70.0, help="Confidence threshold")
    
    args = parser.parse_args()

    try:
        recognizer = FaceRecognizerCUDA(args.model, args.labels)
        
        print(f"Processing image: {args.image}...")
        label, conf, infer_time, probs = recognizer.predict(args.image)
        
        final_result = label if conf > args.threshold else "Unknown Person"

        print("-" * 40)
        print(f"RESULT:        {final_result}")
        print(f"CONFIDENCE:    {conf:.2f}%")
        print(f"GPU TIME:      {infer_time:.2f} ms")
        print("-" * 40)

        print("Class Probabilities:")
        for i, prob in enumerate(probs):
            print(f"  {recognizer.labels[i]:<15}: {prob:.2f}%")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()