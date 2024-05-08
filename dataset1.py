from torch.utils.data.dataset import Dataset
import os
import numpy as np
import torch
from scipy.io import loadmat
import pandas as pd


class MyDataset(Dataset):

    def __init__(self, root_dir, x_dir):
    # def __init__(self, root_dir, x_dir):
        self.root_dir = root_dir
        self.x_dir = x_dir
        self.path = os.path.join(self.root_dir, self.x_dir)
        self.data_path = os.listdir(self.path)

    def __len__(self):
        return len(self.data_path)

    # def __getitem__(self, inx):
    #     data_name = self.data_path[inx]
    #     data_item_path = os.path.join(self.root_dir, self.x_dir, data_name)
    #     data = np.fromfile(data_item_path)
    #     inx = self.x_dir
    #     # return data, inx
    #     return data


    def __getitem__(self, inx):
        data_name = self.data_path[inx]
        data_item_path = os.path.join(self.root_dir, self.x_dir, data_name)
        data = pd.read_csv(data_item_path,names=['num', 'value'])
        v = data['value']
        v = np.array(v)
        v = v.tolist()
        v = torch.unsqueeze(torch.FloatTensor(v), dim=1)
        # data_tensor = torch.from_numpy(data)
        # i =1700 - len(data_tensor)
        # zero = torch.zeros(i)
        # data_tensor_f = torch.cat((data_tensor,zero), 0)
        inx = self.x_dir
        # return data_tensor_f
        # return data_tensor_f, inx
        return v

root_dir = 'E://xhr0//pythonProject4//data'

a_dir = '1R'
b_dir = '1T'
c_dir = '2R'
d_dir = '2T'
e_dir = '3R'
f_dir = '3T'
g_dir = '4R'
h_dir = '4T'
a_dataset = MyDataset(root_dir, a_dir)
b_dataset = MyDataset(root_dir, b_dir)
c_dataset = MyDataset(root_dir, c_dir)
d_dataset = MyDataset(root_dir, d_dir)
e_dataset = MyDataset(root_dir, e_dir)
f_dataset = MyDataset(root_dir, f_dir)
g_dataset = MyDataset(root_dir, g_dir)
h_dataset = MyDataset(root_dir, h_dir)


train_dataset = a_dataset + b_dataset + c_dataset + d_dataset + e_dataset + f_dataset + g_dataset + h_dataset

train_data_size = len(train_dataset)
print("训练数据集的长度为：{}".format(train_data_size))
print(train_dataset[0].shape)
# a_dataset_size = len(a_dataset)
# print(a_dataset_size)
# print(type(a_dataset))
# print(type(a_dataset[1]))
# print(a_dataset[1])
# print(len(a_dataset[1]))
# # # a_dataset = torch.Tensor(a_dataset)
# # print(type(train_dataset[0][0]))
# print(train_dataset[20000][0].shape)
# print(train_dataset[200][0].shape)

# p = pd.read_csv('E://xhr0//pythonProject4//data//1R//1r_1.csv',names=['num', 'value'])
# print(p)
# print(p.shape)
# print(type(p))
# x = p['value']
# print(type(x))  # <class 'pandas.core.series.Series'>
# x = np.array(x)
# print(type(x))  # <class 'numpy.ndarray'>
# x = x.tolist()
# print(type(x))  # <class 'list'>
# x = torch.unsqueeze(torch.FloatTensor(x), dim=1)
# print(type(x))  # <class 'torch.Tensor'>
# print(x.shape)  # torch.Size([97, 1])
# print(x)
