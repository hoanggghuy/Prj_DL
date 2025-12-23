import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os


def main():
    # ================= 1. CẤU HÌNH (CONFIG) =================
    DATA_DIR = './dataset'  # Thư mục chứa ảnh
    MODEL_SAVE_PATH = 'face_model.pth'
    ONNX_SAVE_PATH = 'face_model.onnx'
    LABELS_FILE = 'labels.txt'

    BATCH_SIZE = 32
    NUM_EPOCHS = 200  # Số lần học (tăng lên nếu muốn chính xác hơn)
    LEARNING_RATE = 0.001

    # Kiểm tra xem có GPU không
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy trên thiết bị: {device}")

    # ================= 2. CHUẨN BỊ DỮ LIỆU =================
    if not os.path.exists(DATA_DIR):
        print(f"LỖI: Không tìm thấy thư mục '{DATA_DIR}'. Hãy tạo folder và chép ảnh vào trước.")
        return

    # Các bước xử lý ảnh chuẩn cho MobileNetV2
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Đang load dữ liệu...")
    image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataset_size = len(image_dataset)
    class_names = image_dataset.classes  # Lấy danh sách tên class tự động
    num_classes = len(class_names)

    print(f"Tìm thấy {dataset_size} ảnh.")
    print(f"Tìm thấy {num_classes} người: {class_names}")
    print(f"Map: {image_dataset.class_to_idx}")

    # --- LƯU FILE LABELS.TXT (QUAN TRỌNG CHO JETSON NANO) ---
    print(f"Đang lưu file {LABELS_FILE}...")
    with open(LABELS_FILE, "w") as f:
        for name in class_names:
            f.write(name + "\n")
    # --------------------------------------------------------

    # ================= 3. XÂY DỰNG MODEL =================
    print("Đang tải MobileNetV2 pre-trained...")
    # Tải model MobileNetV2 đã học sẵn trên ImageNet
    model = models.mobilenet_v2(pretrained=True)

    # Đóng băng (Freeze) các lớp đầu để không train lại (giữ đặc trưng cơ bản) - Tùy chọn
    # Nếu dữ liệu ít, nên mở comment dòng dưới để train nhanh hơn và đỡ overfit
    # for param in model.features.parameters():
    #     param.requires_grad = False

    # Thay thế lớp classifier cuối cùng
    # MobileNetV2: classifier[1] là lớp Linear cuối
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)

    model = model.to(device)

    # Định nghĩa hàm Loss và Optimizer
    criterion = nn.CrossEntropyLoss()
    # Dùng SGD với momentum thường tốt cho Fine-tuning
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # ================= 4. TRAINING LOOP =================
    print("Bắt đầu train...")
    model.train()  # Chuyển sang chế độ train

    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Xóa gradient cũ
            optimizer.zero_grad()

            # Forward (Chạy mô hình)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward (Lan truyền ngược & Cập nhật trọng số)
            loss.backward()
            optimizer.step()

            # Thống kê
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = corrects.double() / dataset_size

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

    # ================= 5. LƯU MODEL & XUẤT ONNX =================
    print("-" * 30)

    # 5.1 Lưu file .pth (để dùng lại trên PC hoặc train tiếp)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Đã lưu model PyTorch tại: {MODEL_SAVE_PATH}")

    # 5.2 Xuất file .onnx (để chạy trên Jetson Nano / TensorRT)
    print("Đang xuất sang ONNX cho Jetson Nano...")
    model.eval()  # Chuyển sang chế độ eval trước khi export

    # Tạo input giả lập đúng kích thước
    dummy_input = torch.randn(1, 3, 224, 224, device=device)

    torch.onnx.export(model,
                      dummy_input,
                      ONNX_SAVE_PATH,
                      verbose=False,
                      input_names=['input_0'],
                      output_names=['output_0'],
                      opset_version=11)  # Opset 11 thường ổn định nhất với TensorRT

    print(f"Đã xuất file ONNX tại: {ONNX_SAVE_PATH}")
    print("XONG! Bạn hãy copy 'face_model.onnx' và 'labels.txt' vào Jetson Nano.")


if __name__ == "__main__":
    main()