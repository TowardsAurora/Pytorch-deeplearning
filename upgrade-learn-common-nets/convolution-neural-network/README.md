#### in_channels：输入特征图的通道数。MNIST 是灰度图，故为 1。
#### out_channels：卷积后输出特征图的通道数，决定每层提取多少种特征。
#### kernel_size：卷积核大小，决定每次卷积操作覆盖的区域。
#### stride：卷积核移动步长，影响输出特征图的尺寸。
#### padding：在输入特征图边缘补零，控制输出尺寸，防止特征图变小太快。
#### BatchNorm2d：对每个通道做归一化，加速收敛，提升稳定性。
#### ReLU：激活函数，引入非线性，提高模型表达能力。
#### MaxPool2d：池化层，降低特征图尺寸，减少参数和计算量。
#### Linear：全连接层，将卷积特征展平后用于分类。
#### 计算步骤
输入 输入尺寸：(batch_size, 1, 28, 28)  
### 第一层卷积  
卷积：Conv2d(1, 16, 3, 1, 2) 输出尺寸：(batch_size, 16, 30, 30) 计算公式：output_size = (input_size + 2*padding - kernel_size) / stride + 1
批归一化、ReLU
池化：MaxPool2d(2, 2) 输出尺寸：(batch_size, 16, 15, 15)
### 第二层卷积  
卷积：Conv2d(16, 32, 3, 1, 2) 输出尺寸：(batch_size, 32, 17, 17)
批归一化、ReLU
池化：MaxPool2d(2, 2) 输出尺寸：(batch_size, 32, 8, 8)
展平  
展平成一维：(batch_size, 32*8*8 = 2048)
### 全连接层  
Linear(2048, num_classes) 输出：(batch_size, num_classes)

### eg
假设 batch_size=32，input_size=28*28，输入为 MNIST 灰度图：  
输入层 输入张量形状：(32, 1, 28, 28)  
第一层卷积 卷积参数：in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2 输出尺寸计算： [(28 + 2*2 - 3) / 1 + 1 = 30] 输出形状：(32, 16, 30, 30) 池化后： [30 / 2 = 15] 输出形状：(32, 16, 15, 15)  
第二层卷积 卷积参数：in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=2 输出尺寸： [(15 + 2*2 - 3) / 1 + 1 = 17] 输出形状：(32, 32, 17, 17) 池化后： [17 / 2 = 8.5]，向下取整为 8 输出形状：(32, 32, 8, 8)  
展平 展平后形状：(32, 32*8*8 = 2048)  
全连接层 输入：2048，输出：num_classes=10 输出形状：(32, 10)