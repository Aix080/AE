import scipy.io as sio
import pandas as pd
import numpy as np
import os
import struct
import torch
# Path = 'E://xhr0//pythonProject4//data//4T//C_4t.mat'
Path = ''                       #路径
data = sio.loadmat(Path)

value = data['C']

i = 0

for i in range(10000):
    sig = value[0][i]
    # des_filename = 'E://xhr0//pythonProject4//data//4T//4r'
    des_filename = ''                      #路径
    filename = des_filename + '_' + str(i+1) + '.csv'
    print("第{",i+1,"}个样本")
    data1 = pd.DataFrame(sig)
    data1.to_csv(filename)



# sig = value[0][99999]
# print(len(sig))
# print(sig)
# np.savetxt( 'F://xhr//pythonProject3//data_C//xx.txt',sig)


# # import torch
# #
# # 检查GPU是否可用
# if torch.cuda.is_available():
#     print("GPU可用，深度学习加速之旅开始！")
# else:
#     print("GPU不可用，将使用CPU进行计算。")