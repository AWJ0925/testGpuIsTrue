# 测试GPU是否可用

from torch.utils import data  # 获取迭代数据
from torchvision.datasets import mnist  # 获取数据集
import torch
import torchvision
import time
import matplotlib.pyplot as plt


# 测试GPU是否可用
gpuIs = torch.cuda.is_available()
device = torch.device("cuda:0" if gpuIs else "cpu:0")
print(device)

# 数据集的预处理
data_tf = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),  # 张量
     torchvision.transforms.Normalize([0.5], [0.5])  # 正则化
     ])

data_path = r'./dataset'  # 数据集的相对路径
# 获取数据集
train_data = mnist.MNIST(root=data_path, train=True, transform=data_tf, download=False)
test_data = mnist.MNIST(root=data_path, train=False, transform=data_tf, download=False)
# 打包（分成mini-batch）
train_loader = data.DataLoader(train_data, batch_size=128, shuffle=True, num_workers=0)
test_loader = data.DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)


# 定义网络结构
class CNNnet(torch.nn.Module):

    def __init__(self):
        super(CNNnet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1,
                            out_channels=16,
                            kernel_size=3,
                            stride=2,
                            padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, 2, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, 2, 2, 0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.mlp1 = torch.nn.Linear(2 * 2 * 64, 100)
        self.mlp2 = torch.nn.Linear(100, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0), -1))
        x = self.mlp2(x)
        return x


# Build model
model = CNNnet().to(device)

# Loss and Optimizer
loss_func = torch.nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)  # learning rate

'''-----------------------Training---------------------'''
EPOCHS = 50
loss_count = []
time_start = time.time()
for epoch in range(EPOCHS):
    # 训练阶段
    model.train()
    for batch_id, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        opt.zero_grad()  # 清空上一步残余更新参数值
        # 获取最后输出
        out = model(data).to(device)  # torch.Size([128,10])

        # 获取损失
        loss = loss_func(out, target)
        # 使用优化器优化损失
        loss.backward()  # 误差反向传播，计算参数更新值
        opt.step()  # 将参数更新值施加到 net的 parameters上
        # 每20个mini-batch 统计数据
        if batch_id % 20 == 0:
            loss_count.append(loss)
            print('{}:\t'.format(batch_id), loss.item())
            # torch.save(model,r'C:\Users\liev\Desktop\myproject\yin_test\log_CNN')
        # 每 100个mini-batch 测试网路的性能
        if batch_id % 100 == 0:
            for (data, target) in test_loader:
                # 搬到device上
                test_x = data.to(device)
                test_y = target.to(device)
                out = model(test_x).cpu()
                accuracy = torch.max(out, 1)[1].numpy() == test_y.cpu().numpy()
                print('accuracy:\t', accuracy.mean())
                break

time_end = time.time()
print("CURRENT DEVICE TYPE: {},TOTAL TIME: {:.2f} s.".format(str(device), (time_end - time_start)))

# 绘制损失曲线
plt.figure('PyTorch_CNN_Loss')
plt.plot(loss_count, label='Loss')
plt.legend()
plt.show()


# CURRENT DEVICE TYPE: cpu:0,TOTAL TIME: 171.32 s.
# CURRENT DEVICE TYPE: cuda:0,TOTAL TIME: 101.63 s.
