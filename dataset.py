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
        data = np.fromfile(data_item_path)
        data_tensor = torch.from_numpy(data)
        i =1700 - len(data_tensor)
        zero = torch.zeros(i)
        data_tensor_f = torch.cat((data_tensor,zero), 0)
        inx = self.x_dir
        return data_tensor_f
        # return data_tensor_f, inx

root_dir = 'data'

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
# # # print(type(a_dataset))
# # # print(type(a_dataset[0]))
# # # print(a_dataset[0])
# # # a_dataset = torch.Tensor(a_dataset)
# # print(type(train_dataset[0][0]))
# print(train_dataset[20000][0].shape)
# print(train_dataset[200][0].shape)