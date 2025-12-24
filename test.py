import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class SystemConfig:
    model_path: str = 'face_model.pth'
    labels_path: str = 'labels.txt'
    confidence_threshold: float = 70.0
    input_size: Tuple[int, int] = (224, 224)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class FaceRecognizer:
    def __init__(self, config: SystemConfig):
        self.cfg = config
        print(f"Initializing system on: {self.cfg.device.upper()}")
        
        self.labels = self._load_labels()
        self.model = self._load_model()
        self.transform = self._get_transforms()
        
        self._warmup_gpu()

    def _load_labels(self) -> List[str]:
        if not os.path.exists(self.cfg.labels_path):
            raise FileNotFoundError(f"Missing labels file: {self.cfg.labels_path}")
        with open(self.cfg.labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_model(self) -> nn.Module:
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(f"Missing model file: {self.cfg.model_path}")

        print(f"Loading model weights from {self.cfg.model_path}...")
        try:
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.last_channel, len(self.labels))
            
            state_dict = torch.load(self.cfg.model_path, map_location=self.cfg.device)
            model.load_state_dict(state_dict)
            model.to(self.cfg.device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

    def _get_transforms(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(self.cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _warmup_gpu(self):
        print("Warming up GPU kernels...")
        dummy_input = torch.randn(1, 3, *self.cfg.input_size).to(self.cfg.device)
        with torch.no_grad():
            self.model(dummy_input)
        print("System Ready!")

    def predict(self, image_path: str) -> Tuple[str, float, float]:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.cfg.device)
        except Exception as e:
            raise ValueError(f"Invalid image format: {e}")

        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
            
            if torch.isnan(outputs).any():
                outputs = torch.nan_to_num(outputs, nan=0.0)

            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, idx = torch.max(probs, 0)

        end_time = time.time()
        inference_time_ms = (end_time - start_time) * 1000
        
        confidence_pct = conf.item() * 100
        predicted_label = self.labels[idx.item()]

        return predicted_label, confidence_pct, inference_time_ms

def clean_path(path_str: str) -> str:
    return path_str.strip().strip("'").strip('"')

def main():
    config = SystemConfig()
    
    try:
        engine = FaceRecognizer(config)
    except Exception as e:
        print(f"Critical Error during initialization: {e}")
        sys.exit(1)

    print("\n" + "="*60)
    print("       INTERACTIVE FACE RECOGNITION")
    print("="*60)
    print(f"Classes: {engine.labels}")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input(">> Image Path: ")
            
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("Exiting system...")
                break
            
            if not user_input.strip():
                continue

            clean_input = clean_path(user_input)
            
            try:
                label, conf, t_ms = engine.predict(clean_input)
                
                if conf < config.confidence_threshold:
                    display_label = "UNKNOWN PERSON"
                else:
                    display_label = label

                print("-" * 50)
                print(f"File:       {os.path.basename(clean_input)}")
                print(f"Result:     {display_label}")
                print(f"Confidence: {conf:.2f}%")
                print(f"Latency:    {t_ms:.2f} ms")
                print("-" * 50)

            except FileNotFoundError:
                print("Error: File does not exist.")
            except ValueError as ve:
                print(f"Error: {ve}")

        except KeyboardInterrupt:
            print("\nForced Stop.")
            break

if __name__ == "__main__":
    main()