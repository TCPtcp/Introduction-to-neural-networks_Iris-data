# Introduction-to-neural-networks_Iris-data
本项目是基于B站快速入门pytorch教程自己复现的代码，修正了视频中的部分代码错误，并给出原因说明。  

视频链接：https://www.bilibili.com/video/BV1aT41147p2?p=13&vd_source=88365eb122c512927fdc34b92e0c85d1


1、视频中，train_acc的位置放错了，应该是在一个eopch完成之后再计算train_acc。  

2、评论区也有人指出，该代码的进度条部分实际上是有问题的。我们应该在每个epoch完成之后，打印我们的进度条（从0%-100%），而不是在batch_size上使用进度条。 

解决方法为：先用tqdm对dataloader进行封装，然后再将前面tqdm的返回值传入enumerate中，用以实现每次使用batch_size大小的样本进行训练。

3、train_acc的计算有些地方可能需要修改数据类型。

4、数据加载的时候注意我们的数据格式与视频中数据格式的区别。


大致的问题就上面几个，详细见代码文件data_loader.py和nn.py。
