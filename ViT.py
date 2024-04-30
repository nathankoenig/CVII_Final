import matplotlib.pyplot as plt 
import torch
import torchvision
from torch import nn
from torchvision import transforms, datasets
from torchinfo import summary
import os
from torch.utils.data import DataLoader
import engine

# Seed generator
def set_seeds(seed: int=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# Ensure using GPU (GCP)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Obtain base model's weights and create instance
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# Freeze all blocks (will unfreeze)
for parameter in pretrained_vit.parameters():
	parameter.requires_grad = False

# Unfreeze the last few blocks for fine-tuning
for name, parameter in pretrained_vit.named_parameters():
    if name.startswith('blocks.9') or name.startswith('blocks.10') or name.startswith('blocks.11'):
        parameter.requires_grad = True

# Change class names to match folder
class_names = ['1', '2', '3', '4']

# Set seeds and and create additional training architecture
set_seeds()
pretrained_vit.heads = nn.Sequential(nn.Dropout(0.5), nn.Linear(768, len(class_names))).to(device)

# Set directory paths
train_dir = 'train'
test_dir = 'validation'

# Get transforms
pre = pretrained_vit_weights.transforms()

# Set number of works for optimized training speed
NUM_WORKERS = os.cpu_count()

# Create dataloaders
def create_loaders(train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int, num_workers: int=NUM_WORKERS):
	train_data = datasets.ImageFolder(train_dir, transform=transform)
	test_data = datasets.ImageFolder(test_dir, transform=transform)
	class_names = train_data.classes 
	train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_dataloader, test_dataloader, class_names

# Initialize dataloaders
train_dataloader_pretrained, test_dataloader_pretrained, class_names = create_loaders(train_dir=train_dir, test_dir=test_dir, transform=pre, batch_size=20)

# Create optimizer and loss function
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), lr=1e-3, weight_decay=1e-6)
loss_fn = torch.nn.CrossEntropyLoss()

# Set seeds and train 
set_seeds()
pretrained_vit_results = engine.train(model=pretrained_vit, train_dataloader=train_dataloader_pretrained, test_dataloader=test_dataloader_pretrained, optimizer=optimizer, loss_fn=loss_fn, epochs=250, device=device)

# Train and create plots
def train_model(results):
    loss = results["train_loss"]
    test_loss = results["test_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["test_acc"]
    epochs = range(len(results["train_loss"]))
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="test_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="test_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig('Vit_performance.png')  # Saves the plot as a PNG file


train_model(pretrained_vit_results)



