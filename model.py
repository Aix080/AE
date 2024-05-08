

import torch

from torch import nn
from torch.autograd import Variable

from torchvision.utils import save_image


class ConvAutoencoder(nn.Module):
    def init(self):
        super(ConvAutoencoder, self).init()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # 输入通道数1，输出通道数16
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # 输入通道数16，输出通道数32
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # 输入通道数32，输出通道数64
            nn.ReLU()
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输入通道数64，输出通道数32
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输入通道数32，输出通道数16
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),  # 输入通道数16，输出通道数1
            # nn.Sigmoid()  # 使用Sigmoid激活函数来确保输出值在[0, 1]范围内
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



