import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 3
batch_size = 32
learning_rate = 0.001

# load dataset
transform = transforms.Compose([
    transforms.Pad(4),  # 对图像进行填充操作，在图像的四周添加 4 个像素的边框。这种操作可以增加图像的尺寸，同时保留原始内容。
    transforms.RandomHorizontalFlip(),  #随机水平翻转图像。翻转的概率是 50%，这是一种常见的数据增强方法，用于增加训练数据的多样性。
    transforms.RandomCrop(32),  #从图像中随机裁剪出一个大小为 32x32 的区域。这种操作可以模拟不同的视角或位置变化，从而提高模型的鲁棒性。
    transforms.ToTensor()]  # PIL 图像或 NumPy 数组转换为 PyTorch 的张量格式，同时将像素值归一化到 [0, 1] 的范围内。
)

# CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../basics/data/',
                                                train=True,
                                                transform=transform,
                                                download=True)
val_dataset = torchvision.datasets.CIFAR10(root='../../basics/data/',
                                                train=False,
                                                transform=transform)

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

images, labels = next(iter(val_loader))

print(images[0].shape)
print(images[0].permute(1, 2, 0).numpy())
# images[0].squeeze().numpy() 得到的数组形状是 (3, 32, 32)，
# 这是 RGB 彩色图像的格式（3 个通道）。
# plt.imshow 需要的是 (32, 32, 3) 这种 HWC 格式  Height（高）、Width（宽）、Channel（通道）
print(labels[0].numpy())
plt.imshow(images[0].permute(1, 2, 0).numpy())
# plt.imsave('image.png', images[0].permute(1, 2, 0).numpy())  # label = 3
plt.show()


# build model

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# Residual block  2layers of convolution
class ResidualBlock(nn.Module):
    def __init__(self, in_channels,out_channels,stride = 1,downsample = None) -> None:
        super(ResidualBlock,self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    def forward(self,x):
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        x= self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.relu(x)
        return x


## ResNet build by ResidualBlock
class ResNet(nn.Module):
    def __init__(self, block, layers,num_classes = 10) -> None:
        super(ResNet,self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 64, layers[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride!=1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride=stride, downsample=downsample))
        self.in_channels = out_channels
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


##  Bottleneck 3 layers of convolution
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 第一个1x1卷积，降低维度
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个3x3卷积，处理特征（可能下采样）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 第三个1x1卷积，扩展维度
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
## ResNet build by Bottleneck   this is a resnet example for ImageNet not suitable for CIFAR10
class ResNetBottleneck(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNetBottleneck, self).__init__()
        self.in_channels = 64

        # 初始卷积层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
model2 = ResNetBottleneck(Bottleneck, [2, 2, 2,2])
print("model1")
print(model)
print("model2")
print(model2)
model2.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
def train():
    for epoch in range(num_epochs):
        loop = tqdm(train_loader,total=len(train_loader),desc=f'Epoch [{epoch+1}/{num_epochs}]')
        for i,(images,labels) in enumerate(loop):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())
    # torch.save(model.state_dict(), 'model.pth')

def validate():
    with torch.no_grad():
        model.eval()
        correct = 0
        total = 0
        for images,labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy} %')

def predict(image):
    model.to(device)  # 将模型移动到设备
    model.eval()  # 设置模型为评估模式
    image = image.to(device)
    output = model(image.unsqueeze(0))  # Add batch dimension
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

def predict_using_localmodel(image,path='model.pth'):
    model = ResNet(ResidualBlock, [2, 2, 2])  # 创建模型架构
    model.load_state_dict(torch.load(path, map_location=device))  # 加载权重
    model.to(device)  # 将模型移动到设备
    model.eval()  # 设置模型为评估模式
    image = image.to(device)
    output = model(image.unsqueeze(0))  # Add batch dimension
    _, predicted = torch.max(output.data, 1)
    return predicted.item()

train()
validate()
image_open = Image.open('image.png').convert('RGB')
compose = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
pred = predict(compose(image_open))
print(f'Predicted label: {pred}')  # label = 3


