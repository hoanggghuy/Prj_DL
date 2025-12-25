import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from PIL import Image
import sys
import os
import time

logger = trt.Logger(trt.Logger.ERROR)

class TRTEngine:
    def __init__(self, engine_path):
        self.cuda_ctx = cuda.Device(0).make_context()

        if not os.path.exists(engine_path):
            sys.exit(f"Error: Engine file {engine_path} not found.")

        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        self.inputs = []
        self.outputs = []
        self.allocations = []
        self.input_shape = None

        for i in range(self.engine.num_bindings):
            is_input = self.engine.binding_is_input(i)
            shape = self.engine.get_binding_shape(i)
            dtype = self.engine.get_binding_dtype(i)
            size = trt.volume(shape) * dtype.itemsize
            allocation = cuda.mem_alloc(size)
            
            binding = {
                'index': i,
                'shape': shape,
                'allocation': allocation,
            }
            self.allocations.append(allocation)
            
            if is_input:
                self.inputs.append(binding)
                self.input_shape = shape
            else:
                self.outputs.append(binding)
        
        self._warmup()

    def _preprocess(self, image_path):
        target_h, target_w = self.input_shape[2], self.input_shape[3]
        
        try:
            img = Image.open(image_path).convert('RGB')
            img = img.resize((target_w, target_h), Image.BILINEAR)
            
            img_np = np.array(img).astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            img_np = (img_np - mean) / std
            
            img_np = img_np.transpose(2, 0, 1)
            return np.expand_dims(img_np, axis=0).ravel()
        except Exception:
            return None

    def _warmup(self):
        dummy = np.zeros(self.input_shape, dtype=np.float32).ravel()
        cuda.memcpy_htod(self.inputs[0]['allocation'], dummy)
        self.context.execute_v2(self.allocations)

    def predict(self, image_path):
        start_total = time.time()
        
        input_data = self._preprocess(image_path)
        if input_data is None:
            return None, 0, 0
        
        cuda.memcpy_htod(self.inputs[0]['allocation'], input_data)
        
        t_start_gpu = time.time()
        self.context.execute_v2(self.allocations)
        t_end_gpu = time.time()
        
        output_data = np.zeros(self.outputs[0]['shape'], dtype=np.float32)
        cuda.memcpy_dtoh(output_data, self.outputs[0]['allocation'])
        
        end_total = time.time()
        
        latency_gpu = (t_end_gpu - t_start_gpu) * 1000
        latency_total = (end_total - start_total) * 1000
        
        return output_data.ravel(), latency_gpu, latency_total

    def __del__(self):
        try:
            self.cuda_ctx.pop()
        except:
            pass

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def clean_path(path):
    return path.strip().replace("'", "").replace('"', "")

def main():
    ENGINE_PATH = "face_model.engine"
    LABELS_PATH = "labels.txt"
    
    if not os.path.exists(LABELS_PATH):
        sys.exit("Error: Labels file not found.")

    with open(LABELS_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    print("Initializing TensorRT Engine...")
    try:
        engine = TRTEngine(ENGINE_PATH)
    except Exception as e:
        sys.exit(f"Initialization Failed: {e}")

    print("\n" + "="*50)
    print("TENSORRT INTERACTIVE MODE")
    print("="*50)
    print("Instruction: Drag and drop image file, then press Enter.")
    print("Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            user_input = input(">> Image Path: ")
            
            if user_input.lower() in ['q', 'exit']:
                break
            
            if not user_input.strip():
                continue

            image_path = clean_path(user_input)
            
            if not os.path.exists(image_path):
                print("Error: File does not exist.")
                continue

            raw_output, t_gpu, t_total = engine.predict(image_path)
            
            if raw_output is None:
                print("Error processing image.")
                continue

            probs = softmax(raw_output)
            idx = np.argmax(probs)
            conf = probs[idx] * 100
            label = labels[idx]
            
            final_label = label if conf > 70 else "UNKNOWN"

            print("-" * 45)
            print(f"File:       {os.path.basename(image_path)}")
            print(f"Result:     {final_label}")
            print(f"Confidence: {conf:.2f}%")
            print(f"GPU Time:   {t_gpu:.2f} ms")
            print(f"Total Time: {t_total:.2f} ms")
            print("-" * 45 + "\n")

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()