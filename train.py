import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib.pyplot as plt
import torch.optim as optim
import efficientnet_pytorch
from datetime import datetime
from tqdm import tqdm

class CustomEfficientNet(nn.Module):
    def __init__(self):
        super(CustomEfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
            'efficientnet-b0'
        )
        self.base_model._fc = nn.Linear(
            in_features=self.base_model._fc.in_features, 
            out_features=1, 
            bias=True
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.sigmoid(x)
        return x

config = {
    "model_name": "efficientnet_b0",  # "vit_l_32" or "swin_v2_b" or "efficientnet_b0"
    "criterion": "BCEWithLogitsLoss", # "BCEWithLogitsLoss" or "BCELoss"
    "scheduler": "multistep", # "none" or "exponential" or "multistep"
    "pretrained": True,         # Set False for training from scratch
    "data_root": "./Dataset",
    "batch_size": 16,            # 16
    "num_epochs": 10,            # 10
    "learning_rate": 1e-4,
    "gpus": [0],               # Default is GPU 0, change to [0, 1] for both GPUs
    "output_dir": "./results"
}

if config["model_name"].lower() == "vit_l_32":
    model = models.vit_l_32(weights=models.ViT_L_32_Weights.DEFAULT if config["pretrained"] else None)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, 1)
elif config["model_name"].lower() == "swin_v2_b":
    model = models.swin_v2_b(weights=models.Swin_V2_B_Weights.DEFAULT if config["pretrained"] else None)
    num_features = model.head.in_features
    model.head = nn.Linear(num_features, 1)
elif config["model_name"].lower() == "efficientnet_b0":
    model = CustomEfficientNet()
else:
    raise ValueError("Unsupported model")

# Dataset and DataLoader
transform = transforms.Compose([
    # transforms.Resize((317, 317)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(os.path.join(config["data_root"], "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(config["data_root"], "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

# GPU setup
device = torch.device(f"cuda:{config['gpus'][0]}" if torch.cuda.is_available() else "cpu")
model.to(device)

if len(config["gpus"]) > 1:
    model = nn.DataParallel(model, device_ids=config["gpus"])

def get_criterion():
    if config[criterion].lower() == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss()
    elif config[criterion].lower() == "bceloss":
        return nn.BCELoss()
    else:
        raise ValueError("Unsupported loss function")

# Loss and Optimizer
criterion = get_criterion()
optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
scaler = GradScaler()

# Get scheduler for dynamic learning_rate
def get_scheduler():
    if config["scheduler"] == "none":
        return None
    elif config["scheduler"] == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.5) # lr = lr*gamma**epoch
    elif config["scheduler"] == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,8], gamma=0.1)
    else:
        raise ValueError("Unsupported scheduler")

# Training and Validation Function
def train(epoch):
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]", leave=False)
    scheduler = get_scheduler()
    
    for i, (images, labels) in enumerate(progress_bar):
        images, labels = images.to(device), labels.to(device).float()
        labels = labels.unsqueeze(1)
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        predictions = torch.sigmoid(outputs) > 0.5
        total_correct += predictions.eq(labels).sum().item()
        total_samples += labels.size(0)
        loss=total_loss/(total_samples/config["batch_size"])
        acc=total_correct/total_samples
        progress_bar.set_postfix(loss=loss, acc=acc)

    # Update scheduler
    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / len(train_loader)
    avg_acc = total_correct / total_samples
    print(f"Train Epoch: {epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

def validate(epoch):
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [VALID]", leave=False)
    with torch.no_grad():
        for i, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(device), labels.to(device).float()
            labels = labels.unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predictions = torch.sigmoid(outputs) > 0.5
            total_correct += predictions.eq(labels).sum().item()
            total_samples += labels.size(0)
            loss=total_loss/(total_samples/config["batch_size"])
            acc=total_correct/total_samples
            progress_bar.set_postfix(loss=loss, acc=acc)


    avg_loss = total_loss / len(val_loader)
    avg_acc = total_correct / total_samples
    print(f"Val Epoch: {epoch} Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}")
    return avg_loss, avg_acc

# Logging Setup
train_name = config["model_name"]+'-'+datetime.now().strftime("%Y%m%d%H%M%S")
output_folder = os.path.join(config["output_dir"], train_name)
os.makedirs(output_folder, exist_ok=True)
log_file = os.path.join(output_folder, f"{train_name}-training_log.txt")

with open(log_file, "w") as f:
    f.write("Training Configuration:\n")
    for key, value in config.items():
        f.write(f"{key}: {value}\n")
    f.write("\n")

# Training Loop
train_losses, train_accs, val_losses, val_accs = [], [], [], []
for epoch in range(config["num_epochs"]):
    train_loss, train_acc = train(epoch)
    val_loss, val_acc = validate(epoch)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    print(f"Epoch {epoch}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}\n")
    # Logging
    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Train Loss: {train_loss}, Train Acc: {train_acc}, Val Loss: {val_loss}, Val Acc: {val_acc}\n")
        
model_save_path = os.path.join(output_folder, f"{train_name}.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Plotting
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Training and Validation Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title("Training and Validation Accuracy")
plt.legend()

plt.savefig(os.path.join(output_folder, "{train_name}-metrics.png"))
plt.show()
