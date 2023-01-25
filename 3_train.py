import torch
import torch.utils.data as Data
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np

from torch import nn
from d2l import torch as d2l

import os
import cv2

data_dir = r'C:\Users\lenovo\Desktop\MyCode\FinalData\LGE'

def searchPath(filename):
#寻找文件filename，如果不存在则在当前路径创建新文件
    if not os.path.isdir(filename):
        os.makedirs(filename)
    return

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

BATCH_SIZE = 8

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
    x: ImageFolder(os.path.join('/content/drive/MyDrive/Hello_Colab/Data/LGE', x), transformsDir[x]) #对train,validation,test图片变换字典的读取
    for x in ['train', 'validation', 'test']
}

loader = {
    x: Data.DataLoader(dataset=dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
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

train_iter = loader['train']
valid_iter = loader['validation']
test_iter = loader['test']

lr = 0.01
num_epochs = 40
d2l.train_ch6(net, train_iter, valid_iter, num_epochs, lr, d2l.try_gpu(0))

d2l.plt.show()
# torch.save(net.state_dict(), 'Ori_C0.pth')


"""
数据预测，按批次打印输出结果并且将结果导入到txt文件中
"""
def evaluate_accuracy_gpu(net, data_iter, device="cuda"):
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

def myPredict(net, test_iter, device="cuda", batch_size=BATCH_SIZE):
    """预测标签"""
    # net.apply(init_weights) #参数初始化(预测时需要使用之前训练的模型)
    print('training on', device)  # 打印当前训练的设备，device作为传入参数
    net.to(device)  # 依次将网络和数据搬移到GPU上

    # 调用函数，打印预测精度
    print(evaluate_accuracy_gpu(net, test_iter))

    batch_index = 0
    for X, trues in test_iter:  # X为3*255*255待预测图片，trues为实际标签
        size = X.shape  # batch_size*3*256*256
        X = X.to(device)
        preds = torch.zeros(batch_size)
        tempList=['0',"0","0","0","0","0","0","0","0"]
        tempList[0] = "batch_index: "+str(batch_index)+"\n" #打印第一行，反应批次
        for num in range(batch_size):
          # 逐3D图片预测，net要求输入为四维需要进行维度扩展(也可以使用.unsqueeze()方法)
          # 需要argmax输出为10个类别对应的sigmoid值，并使用item()属性精简输出结果
          temp =net(X[num, :, :, :].reshape(1, size[1], size[2], size[3])).argmax().item()
          preds[num] = temp
          tempList[num+1] = str(trues[num].item()==temp)+" true_label:"+str(temp)+"  "+"pred_label:"+str(trues[num].item())+"\n"

        print("batch_index:{}, true_label:{}, predict_label:{}".format(batch_index, trues, preds))  # 输出每个batch预测结果
        if batch_index==0:
            writeMode = "w" #"w"表示覆盖读写
        else:
            writeMode = "a" #"a"表示不覆盖续写
        searchPath("/content/drive/MyDrive/Hello_Colab/textResult.txt")
        with open("/content/drive/MyDrive/Hello_Colab/textResult.txt",writeMode) as f:    #写入列表每一行
          f.writelines(tempList) #使用writelines将列表的每一项元素写入txt文件中，按行写入需要给元素末尾手动加入\n，保证每个元素代表着文件中一行字符
        batch_index += 1


myPredict(net, test_iter)

"""
# d2l.train_ch6的源代码
def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型(在第六章定义)"""
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    print('training on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
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
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')
"""