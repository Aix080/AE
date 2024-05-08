import torch
import torch.nn as nn


# PyTorch中的所有神经网络模型都应该继承自nn.Module基类，并进行初始化。
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # # 编码器部分，高维->低维。
        self.encoder = nn.Sequential(
            nn.Linear(1700, 512),  # 第一层全连接层，将输入的1700维数据压缩到512维。
            nn.ReLU(),  # 激活函数ReLU，用于增加网络的非线性，帮助模型学习复杂的特征。
            nn.Linear(512, 128),  # 第二层全连接层，进一步将数据从512维压缩到128维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(128, 64),  # 第三层全连接层，将数据从128维压缩到64维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(64, 12),
            nn.ReLU(),
            nn.Linear(12, 3)  # 最后一层全连接层，将数据最终压缩到3维，得到编码后的数据。
         )
        # 解码器部分，低维->高维。
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),  # 第一层全连接层，将编码后的3维数据扩展到12维。
            nn.ReLU(),  # 使用ReLU激活函数。
            nn.Linear(12, 64),  # 第二层全连接层，将数据从12维扩展到64维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(64, 128),  # 第三层全连接层，将数据从64维扩展到128维。
            nn.ReLU(),  # 再次使用ReLU激活函数。
            nn.Linear(128,512),  # 将数据从128维扩展回784维，即原始大小。
            nn.ReLU(),
            nn.Linear(512,1700),
            nn.ReLU()
         )

    def forward(self, x):
        x = x.to(torch.float32)
        x = self.encoder(x)  # 将输入数据通过编码器压缩。
        x = self.decoder(x)  # 然后通过解码器进行重构。
        return x  # 返回重构的数据