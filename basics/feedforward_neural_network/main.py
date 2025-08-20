import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
# Hyper-parameters
input_size = 28 * 28
hiden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 32
learning_rate = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# load MNIST dataset
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


class FeedForwardNet(nn.Module):
    def __init__(self,input_size,hiden_size,num_classes) :
        super(FeedForwardNet,self).__init__()
        self.fc1 = nn.Linear(input_size,hiden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hiden_size,num_classes)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Feed_Forward_model
model = FeedForwardNet(input_size,hiden_size,num_classes)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


"""
除了 .squeeze()，还有一些常用的方法可以处理张量的维度：  
.unsqueeze(dim)：在指定位置增加一个维度，比如把 [28, 28] 变成 [1, 28, 28]。
.view() 或 .reshape()：可以重新调整张量的形状，比如 images.view(-1, 28*28)。
.permute()：可以交换维度的顺序，比如 images.permute(1, 2, 0)。
.flatten()：将多维张量展平成一维
"""

images, labels = next(iter(train_loader))
# print(images[0],labels[0])
plt.imshow(images[0].squeeze(),cmap='gray')
# plt.savefig('image.png')
plt.show()

# evaluate the model
def accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        # reshape images to (batch_size, input_size)
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predict = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predict == labels).sum()
    return correct / total

# train the model
total_steps = len(train_loader)
def train(num_epochs):
    model.to(device)
    for epoch in range(num_epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for i,(images,labels) in loop:
            images = images.reshape(-1, input_size).to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())


train(num_epochs)
acc = accuracy(val_loader)
print(f'Accuracy: {acc}')


def predict(image):
    model.eval()
    image = image.reshape(-1, input_size).to(device)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    return predicted.item()


image_open = Image.open('image.png')
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,))
])
print(image_open)
transform_image = transform(image_open)
print(transform_image)
print("predict:",predict(transform_image))
