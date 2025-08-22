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
input_size = 28
hidden_size = 128
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

# define  Recurrent neural network

class  EASYRNN(nn.Module):

    def __init__(self, input_size,hidden_size,num_classes) -> None:
        super(EASYRNN,self).__init__()
        self.rnn = nn.RNN(input_size,hidden_size,num_classes,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        # x shape: (batch_size, seq_length, input_size)
        print(x.shape)
        x = x.reshape(-1, 28, 28)  # Reshape to (batch_size, seq_length, input_size)
        print(x.shape)
        x, _ = self.rnn(x)
        # out shape: (batch_size, seq_length, hidden_size)
        x =self.fc(x[:, -1, :])
        print(x.shape)
        return x

class RNN_LSTM(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes) -> None:
        super(RNN_LSTM,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size,num_classes)
    def forward(self,x):
        # print(x.shape)
        x = x.reshape(-1, 28, 28)
        # print(x.shape)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        x,_ = self.lstm(x,(h0,c0))
        x= self.fc(x[:,-1,:])
        return x

# model = EASYRNN(input_size,hidden_size,num_classes).to(device)
model = RNN_LSTM(input_size,hidden_size,num_layers=2,num_classes=num_classes).to(device)

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
# image.show("image")
compose = transforms.Compose(
    [transforms.Grayscale(num_output_channels=1), transforms.Resize((28, 28)), transforms.ToTensor(), ])

print(predict(compose(image)))