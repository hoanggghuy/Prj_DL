import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import os
import sys


def main():
    DATA_DIR = './dataset'
    MODEL_SAVE_PATH = 'face_model.pth'
    CHECKPOINT_PATH = 'checkpoint.pth'
    ONNX_SAVE_PATH = 'face_model.onnx'
    LABELS_FILE = 'labels.txt'
    LOG_DIR = 'runs/face_experiment'

    BATCH_SIZE = 32
    NUM_EPOCHS = 400
    LEARNING_RATE = 0.001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    if not os.path.exists(DATA_DIR):
        print(f"Error: Directory '{DATA_DIR}' not found.")
        return

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print("Loading data...")
    image_dataset = datasets.ImageFolder(DATA_DIR, data_transforms)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

    dataset_size = len(image_dataset)
    class_names = image_dataset.classes
    num_classes = len(class_names)

    print(f"Found {dataset_size} images.")
    print(f"Found {num_classes} classes: {class_names}")

    with open(LABELS_FILE, "w") as f:
        for name in class_names:
            f.write(name + "\n")

    print("Loading MobileNetV2...")
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    writer = SummaryWriter(LOG_DIR)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Checkpoint found at '{CHECKPOINT_PATH}'. Loading...")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        print(f"Resumed from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")

    if start_epoch >= NUM_EPOCHS:
        print(f"Training already finished ({start_epoch}/{NUM_EPOCHS}).")
        print("To continue training, please increase NUM_EPOCHS in the code.")
        return

    print("Starting training...")
    model.train()

    for epoch in range(start_epoch, NUM_EPOCHS):
        running_loss = 0.0
        corrects = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_size
        epoch_acc = corrects.double() / dataset_size

        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        torch.save(checkpoint, CHECKPOINT_PATH)

    print("-" * 30)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved at: {MODEL_SAVE_PATH}")
    print(f"Checkpoint saved at: {CHECKPOINT_PATH}")

    writer.close()

    print("Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(model, dummy_input, ONNX_SAVE_PATH, verbose=False, opset_version=11)
    print(f"ONNX file exported at: {ONNX_SAVE_PATH}")


if __name__ == "__main__":
    main()