# 这个代码主要是用于辅助我们去查看读取数据的格式问题，查看读取的格式是否正确
import pandas as pd
import numpy as np

df = pd.read_csv('./Iris数据集/iris.txt')
print(df.head())