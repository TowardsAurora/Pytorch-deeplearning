import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 28*28
num_epochs = 10
num_classes = 10
batch_size = 32
learning_rate = 0.0005

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(
    root='../mnist',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

val_dataset = torchvision.datasets.MNIST(
    root='../mnist',
    train=False,
    transform=transforms.ToTensor(),
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# define  Convolutional neural network

class CNN(nn.Module):
    def __init__(self, num_classes) -> None:
        super(CNN,self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2), # (32,28,28)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc3 = nn.Linear(in_features=int(input_size/16)*64,out_features=num_classes)

    def forward(self,x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.reshape(x.size(0),-1)
        x = self.fc3(x)
        return x

model = CNN(num_classes).to(device)

# define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train(num_epochs):
    for epoch in range(num_epochs):
        model.train(mode=True)
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}')
        for i,(images,labels) in loop:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch {epoch + 1}/{num_epochs}')
            loop.set_postfix(loss=loss.item())

def evaluate(val_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

def predict(image):
    model.eval()
    image = image.reshape(1,1,28,28).to(device)   ## 第一层是 Conv2d，需要输入形状为 [batch_size, 1, 28, 28]。
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

train(num_epochs)
acc = evaluate(val_loader)
print(f'Accuracy of the model on the test images: {acc} %')

from PIL import Image
image = Image.open('image1.png')
image.show("image")
compose = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28)), transforms.ToTensor(), ])

print(predict(compose(image)))