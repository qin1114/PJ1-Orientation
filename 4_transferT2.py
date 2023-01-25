"""
将LGE数据集上的训练结果存储在本地rar文件，读取后迁移到T2数据集上
在T2数据集上：先冻结除了最后一层64*8网络以外的所有参数训练10epoch，再整体训练5epoch
对d2l.train_c6函数进行改进，将上述训练网络合并为trainMixed函数
"""
import torch
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn

import numpy as np
import cv2
import os

#os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #否则内存会爆

BATCH_SIZE = 8

class AddChannel(object):
    """
    增加截断后的图片[0.6、0.8、1]为三通道，并且进行直方图均衡化
    由于不传入新的参数，可以直接调用__call__方法
    """
    def __call__(self, img):
        #传入和传出都要求是PTL格式
        #thres = [0.8, 0.6]
        img_=np.asarray(img).copy().astype(np.uint8) #转为cv2可处理格式(深复制copy后对象完全独立)，并且转化为无符号整型
        maxGray = img_.max().max() #灰度值范围是0~255
        temp = cv2.equalizeHist(img_)
        #产生截断后的3D图片，转化为uint8才能直方图均衡化(threshold返回的是元组(size, data)，要改变类型)
       #     print(cv2.threshold(img_, maxGray, maxGray, 0)[1].shape)
        maxGray60 = (maxGray*0.6).astype(np.uint8)
        temp60 = cv2.threshold(img_, maxGray60, maxGray60, 0)[1]
        temp60 = cv2.threshold(img_, maxGray60, maxGray60, 0)[1]
        maxGray80 = (maxGray * 0.8).astype(np.uint8)
        temp80 = cv2.threshold(img_, maxGray80, maxGray80, 0)[1]
        temp80 = cv2.threshold(img_, maxGray80, maxGray80, 0)[1]
        new_image = np.stack((temp, cv2.equalizeHist(temp80), cv2.equalizeHist(temp60)),axis=2) #产直方图均衡化
  #      print(new_image.shape)
        return new_image


def trainMixed(net, train_iter, test_iter, stage_size, num_epochs, lr, grad_list, device, flag=False):
    """
    在d2l.train_ch6基础上用GPU训练模型:先冻结参数再解冻，num_epochs和lr和grad_list为二维元组，stage_size反应训练阶段数
    """

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    if flag:  # flag=True表明未进行参数迁移，需要进行初始化
        net.apply(init_weights)  # 仅作用于梯度需要更新时(一般默认不随机初始化参数)

    print('training on', device)
    net.to(device)
    # filter(lambda p: p.requires_grad, model.parameters())
    loss = nn.CrossEntropyLoss()
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, sum(num_epochs)],  # 注意修正横坐标长度
                            legend=['train loss', 'train acc', 'test acc'])

    for j in range(stage_size):
        for p in net[:-1].parameters():  # 访问除了最后一层外的所有参数
            p.requires_grad = grad_list[j]  # 冻结网络参数后再解冻
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=lr[j])  # 修正优化器，使得False可以被接受

        for epoch in range(num_epochs[j]):
            xAxis = sum(num_epochs[0:j])  # 计算横坐标的起点：左包含右不包含
            # 训练损失之和，训练准确率之和，样本数
            metric = d2l.Accumulator(3)
            net.train()
            for i, (X, y) in enumerate(train_iter):
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                l = loss(y_hat, y)
                l.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                    animator.add(xAxis + epoch + (i + 1) / num_batches, (train_l, train_acc, None))
            test_acc = evaluate_accuracy_gpu(net, test_iter)
            animator.add(xAxis + epoch + 1, (None, None, test_acc))
        print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
              f'test acc {test_acc:.3f}')
        print(f'{metric[2] * num_epochs[j] / timer.sum():.1f} examples/sec '
              f'on {str(device)}')


def evaluate_accuracy_gpu(net, data_iter, device="cpu"):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def myPredict(net, test_iter, device="cpu", batch_size=BATCH_SIZE):
    """预测标签"""
    # net.apply(init_weights) #参数初始化(预测时需要使用之前训练的模型)
    print('training on', device)  # 打印当前训练的设备，device作为传入参数
    net.to(device)  # 依次将网络和数据搬移到GPU上

    # 调用函数，打印预测精度
    print(evaluate_accuracy_gpu(net, test_iter))



transformsDir = { #存储transform的字典：键：数据集名称；值：对应数据集的transform
    'train': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        AddChannel(), #暂时不使用外来传入参数
        transforms.ToTensor()
    ]),
    'validation': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        AddChannel(), #暂时不使用外来传入参数
        # transforms.Resize((256, 256)),
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
  #      transforms.RandomRotation(10),
   #     transforms.RandomResizedCrop((256, 256), scale=(0.7, 1), ratio=(0.8, 1.2)),
        transforms.Resize((256, 256)),
        AddChannel(), #暂时不使用外来传入参数
        transforms.ToTensor()
    ])
}


#dataset = ImageFolder('/content/drive/MyDrive/Hello_Colab/Data/LGE', transforms)
dataset = {  #同时读取LGE目录下的三个数据夹
    x: ImageFolder(os.path.join(r'.\FinalData\T2', x), transformsDir[x]) #对train,validation,test图片变换字典的读取
    for x in ['train', 'validation', 'test']
}

loader = {
    x: Data.DataLoader(dataset=dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    for x in ['train', 'validation', 'test']
}

net = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(32, 64, kernel_size=3, padding=1),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(64 * 16 * 16, 64), nn.Sigmoid(),
    nn.Linear(64, 8)
)

#从本地文件中载入LGE上预训练的网络参数
net.load_state_dict(torch.load(r"C:\Users\lenovo\Desktop\MyCode\Result\saveNet.tar", map_location=torch.device('cpu')))

train_iter = loader['train']
valid_iter = loader['validation']
test_iter = loader['test']


#myPredict(net, test_loader) #代入测试集预测，判断是否加载成功参数
num_epochs = [10, 5]
lr = [0.001, 0.01]
grad_list = [False, True]
# 先冻结训练参数10次，再解冻继续训练5次
trainMixed(net, train_iter, test_iter, 2, num_epochs, lr, grad_list, d2l.try_gpu(0))




"""
# 保存网络中的参数, 速度快，占空间少
torch.save(net.state_dict(),"/content/drive/MyDrive/Hello_Colab/saveNet.tar")

def init_weights(m): #重新随机初始化权重参数
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
net.apply(init_weights)
myPredict(net, test_iter) #代入测试集预测，判断是否加载成功参数，可以发现准确性为0.125，相当于1/8
#print(net[0].weight) #返回网络第一层权重

net.load_state_dict(torch.load("/content/drive/MyDrive/Hello_Colab/saveNet.tar")) #载入网络参数
myPredict(net, test_iter) #代入测试集预测，加载成功参数
#print(net[0].weight)

# 保存整个网络
torch.save(net, "/content/drive/MyDrive/Hello_Colab/saveNet1.tar")
net1 = torch.load("/content/drive/MyDrive/Hello_Colab/saveNet1.tar") #加载出完整网络
print(net1) #打印网络结构
myPredict(net1, test_iter) #代入测试集预测，判断是否加载成功参数

"""
