import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset import train_dataset
from AE_model import Autoencoder

# 如果有GPU可用，优先使用GPU, 否则使用CPU运算
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')

train_data_size = len(train_dataset)
print("训练数据集的长度为：{}".format(train_data_size))

train_dataset = train_dataset
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=False)

# data = train_dataset
# print(type(data[32]))
# print(data[32])

model = Autoencoder()  # 实例化自编码器（AE）模型
model.to(device)  # 如果cuda可用，则将模型从CPU移动到GPU上进行计算

criterion = nn.MSELoss()  # 损失函数
optimizer = optim.Adam(model.parameters(), lr=1e-3)  # 优化器，学习率为0.001

num_epochs = 20  # 训练周期
lowest_loss = float('inf')  # 初始化最低损失为正无穷，用于跟踪保存最好的模型
# i=0
#
# for data in train_dataloader:
#     i = i + 1
#     value = data
#     value = value.to(device)  # 将数据从CPU移动到指定的设备
#     print(value.shape)
#     print(i)


for epoch in range(num_epochs):
    total_loss = 0.0
    for data in train_dataloader:
        # 自编码器只要数据，不需要标签.
        value = data
        value = value.to(torch.float32)
        value = value.to(device)  # 将数据从CPU移动到指定的设备
        print(value.shape)
        print(value)
        output = model(value)  # 通过模型前向传播得到重建的输出
        loss = criterion(output, value)  # 计算重建数据与原数据之间的损失

        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播，计算梯度
        optimizer.step()  # 根据梯度更新模型参数
        # print(epoch)
        total_loss += loss.item()  # 累加损失
    # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')
    avg_loss = total_loss / len(train_dataloader)  # 计算这个epoch的平均损失
    print(avg_loss)
    print(epoch)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

    # 如果损失是目前为止最低的，保存模型
    # if avg_loss < lowest_loss:
    #     lowest_loss = avg_loss
    #     torch.save(model.state_dict(), 'AEbest.pth')  # 保存模型
    #     print(f'New lowest average loss {lowest_loss:.4f} at epoch {epoch + 1}, model saved.')