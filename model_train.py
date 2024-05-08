from torch.utils.data import DataLoader
from dataset import train_dataset
from model import *


# 检查是否有CUDA支持的GPU可用，如果有，则使用第一个GPU；否则，使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_data_size = len(train_dataset)
print("训练数据集的长度为：{}".format(train_data_size))
# print(train_dataset[0])
# b = train_dataset[0][1]
# print(b)
train_dataset = train_dataset
train_dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0, drop_last=False)
value = train_dataset
# print(len(value))
# print(value[0])
# print(label)
#

i = 0
for data in train_dataloader:
    value = data
    value = value.to(device)
    # label = label.cuda()
    i = i +1
    print(i)
    print(value.shape)
    # print(label)

#
# mymodule = autoencoder().cuda()
#
# loss_function = nn.CrossEntropyLoss()
# loss_function = loss_function.cuda()
#
# learning_rate = 0.001
# optimizer = torch.optim.SGD(mymodule.parameters(), lr=learning_rate)
# epochs = 1000
# for i in range(epochs):
#     print("第{}轮训练开始".format(i+1))
#     for data in train_dataloader:
#         # inputs = data
#         inputs = data.cuda()
#         outputs = mymodule(inputs)
#         loss = loss_function(outputs)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_train_step = total_train_step + 1
#         print("训练次数：{}".format(total_train_step, loss.item()))