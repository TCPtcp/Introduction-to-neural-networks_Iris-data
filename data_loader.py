# 导入库
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch


class iris_dataloader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path  # 数据集路径

        assert os.path.exists(self.data_path), "dataset path does not exist"  # 判断数据集是否存在

        df = pd.read_csv(self.data_path, names = [0, 1, 2, 3, 4])  # 读取数据集

        d = {"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}  # 标签映射
        df[4] = df[4].map(d)  # 标签映射

        data = df.iloc[:,:4]  # 选取前四列作为数据
        label = df.iloc[:, 4]

        data = (data - np.mean(data)/np.std(data))  # 数值归一化  Z值化
        # 这里注意np.mean的大小，这里是不是也用到了广播机制？
        self.data = torch.from_numpy(np.array(data, dtype = "float32"))  # 将数据从dataframe——>numpy——>tensor
        self.label = torch.from_numpy(np.array(label, dtype = "int64"))  # 将标签从dataframe——>numpy——>tensor
        
        self.data_num = len(label)  # 数据集大小
        print("当前数据集大小为：", self.data_num)
    
    def __len__(self):  # 获取数据集大小  # __len__和__getitem__是Dataset的两个必须方法
        return self.data_num  # 返回数据集大小
    
    def __getitem__(self, index):  # 通过索引获取数据及其对应标签
        self.data = list(self.data)
        self.label = list(self.label)

        return self.data[index], self.label[index]  # 返回一一对应的数据和标签