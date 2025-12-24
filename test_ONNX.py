import onnxruntime as ort
import numpy as np
from PIL import Image
import os
import sys
import time
import argparse
from typing import List, Tuple, Optional

class FaceRecognizer:
    def __init__(self, model_path: str, labels_path: str):
        self.model_path = model_path
        self.labels_path = labels_path
        self.labels = self._load_labels()
        self.session = self._load_model()
        self.input_name = self.session.get_inputs()[0].name
        
        # Mean and Std for ImageNet normalization
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _load_labels(self) -> List[str]:
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")
        with open(self.labels_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_model(self) -> ort.InferenceSession:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        # Check available providers (CUDA or CPU)
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"Loading model: {self.model_path}")
        t_start = time.time()
        session = ort.InferenceSession(self.model_path, providers=providers)
        print(f"-> Model loaded in {(time.time() - t_start) * 1000:.2f} ms")
        return session

    def _preprocess(self, image_path: str) -> np.ndarray:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        
        img_data = np.array(img).astype('float32') / 255.0
        img_data = (img_data - self.mean) / self.std
        
        # HWC to CHW and add batch dimension
        img_data = img_data.transpose(2, 0, 1)
        img_data = np.expand_dims(img_data, axis=0)
        return img_data

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def predict(self, image_path: str) -> Tuple[str, float, float, np.ndarray]:
        input_data = self._preprocess(image_path)

        # Warm-up run
        self.session.run(None, {self.input_name: input_data})

        # Inference run
        start_time = time.time()
        outputs = self.session.run(None, {self.input_name: input_data})
        end_time = time.time()

        inference_time_ms = (end_time - start_time) * 1000
        
        raw_scores = outputs[0][0]
        probs = self._softmax(raw_scores)
        
        top_idx = np.argmax(probs)
        confidence = probs[top_idx] * 100
        predicted_label = self.labels[top_idx]

        return predicted_label, confidence, inference_time_ms, probs

def main():
    parser = argparse.ArgumentParser(description="ONNX Face Recognition Inference")
    parser.add_argument("--image", type=str, default="1.jpg", help="Path to input image")
    parser.add_argument("--model", type=str, default="face_model.onnx", help="Path to ONNX model")
    parser.add_argument("--labels", type=str, default="labels.txt", help="Path to labels file")
    parser.add_argument("--threshold", type=float, default=70.0, help="Confidence threshold")
    
    args = parser.parse_args()

    try:
        recognizer = FaceRecognizer(args.model, args.labels)
        
        print(f"Processing image: {args.image}...")
        label, conf, infer_time, probs = recognizer.predict(args.image)
        
        fps = 1000.0 / infer_time if infer_time > 0 else 0
        final_result = label if conf > args.threshold else "Unknown Person"

        print("-" * 40)
        print(f"RESULT:        {final_result}")
        print(f"CONFIDENCE:    {conf:.2f}%")
        print(f"INFERENCE:     {infer_time:.2f} ms")
        print(f"ESTIMATED FPS: {fps:.2f}")
        print("-" * 40)

        print("Class Probabilities:")
        for i, class_name in enumerate(recognizer.labels):
            print(f"  {class_name:<15}: {probs[i]*100:.2f}%")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()