import os
import sys
import torch.utils
from torch.utils.data import DataLoader  # dataloader是什么？
from tqdm import tqdm  

import torch
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化器

from data_loader import iris_dataloader  # 导入数据加载模块


# 初始化神经网络模型

class NN(nn.Module):  # 未完待续
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):  #定义网络层级结构
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim1)  # 输入层
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)  # 隐藏层
        self.layer3 = nn.Linear(hidden_dim2, out_dim)  # 输出层

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x 

# 定义计算环境
# 判断是否有GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # python三模运算

# 数据集的划分和加载。训练集，验证机和测试集
custom_dataset = iris_dataloader('./Iris数据集/iris.data.txt')  # 数据加载
train_size = int(len(custom_dataset) * 0.7)
val_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - train_size - val_size

# random_split函数接收两个参数：数据集、切分比例
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])  # 随机划分数据集

train_Loader = DataLoader(train_dataset, batch_size=16, shuffle=True)  # 数据加载器
# batch_size: 每次读取的数据量，shuffle: 是否打乱数据集
# DataLoader是一个可迭代对象，可以使用for循环读取数据
val_Loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
test_Loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 打印数据集的一些信息
print('训练集的大小train size:', len(train_dataset), 
        '验证集的大小val size:', len(val_dataset), 
        '测试集的大小test size:', len(test_dataset))


# 定义一个推理函数，用来计算并返回准确率。

def infer(model, dataset, device):
    model.eval()  # 模型进入推理模式
    acc_num = 0
    with torch.no_grad():  # 不需要计算梯度
        for data in dataset:
            inputs, labels = data  # 在data_Loader中的item()方法中定义的
            outputs = model(inputs.to(device))
            labels = labels.to(device)
            predicted = torch.max(outputs.data, 1)[1] # 取出所有可能预测类别的最大值，作为预测结果
            acc_num += (predicted == labels).sum().item()  # 这里使用的是item迭代器，也可以写成if语句
            # acc_num += torch.eq(predicted, labels).sum().item()  # 这里使用的是torch中的equal（相等）函数，与上面命令等价

    acc = acc_num / len(dataset)
    return acc

def main(lr = 0.005, epochs = 20):
    # 初始化模型
    model = NN(4, 100, 50, 3).to(device)  # 初始化模型
    loss_f = nn.CrossEntropyLoss()  # 定义损失函数（使用的是交叉熵损失）

    pg = [p for p in model.parameters() if p.requires_grad]  # 获取模型参数
    optimizer = optim.Adam(pg, lr=lr)  # 定义优化器

    # 权重存储路径
    save_path = os.path.join(os.getcwd(), "results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    
    # 开始训练模型
    for epoch in range(epochs):
        model.train()  # 模型进入训练模式
        acc_num = torch.zeros(1)
        sample_num = 0

        train_bar = tqdm(train_Loader, file=sys.stdout, ncols=100)  # 进度条
        for batch_idx, (inputs, labels) in enumerate(train_bar):  # tqdm是一个进度条库
            labels = labels.squeeze(-1)  # 清除掉标签多余的维度
            sample_num += inputs.shape[0]
            optimizer.zero_grad()  # 梯度清零  # 小技巧：前向传播前，梯度清零。即梯度清零命令放在前向传播前比较好
            outputs = model(inputs.to(device))  # 前向传播
            labels = labels.to(device)
            pred_class = torch.max(outputs, dim = 1)[1]
            # torch.max返回的是一个元组，第一个元素是max的值，第二个元素是max值的索引
            acc_num += torch.eq(pred_class, labels).sum() # 计算准确个数
            loss = loss_f(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播 // 求导
            optimizer.step()  # 更新参数

        train_acc = float(acc_num / sample_num)
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)
        # train_bar是含进度条的东西
 
        # 验证模型
        # train_acc = infer(model, train_Loader, device)
        val_acc = infer(model, val_Loader, device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc{:.3f} val_acc{:.3f}".format(epoch+1, epochs, loss, train_acc, val_acc))
        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))  # 存储模型参数

        # 每次数据集迭代一遍（一个eopch）之后，要对初始化的指标清零
        train_acc = 0
        val_acc = 0

    print("Finished Training")

    # 测试模型
    test_acc = infer(model, test_Loader, device)
    print('test acc:', test_acc)

if __name__ == "__main__":
    main(lr=0.005)  # 或者直接写为main()，因为我们已经设置了默认的初始化参数