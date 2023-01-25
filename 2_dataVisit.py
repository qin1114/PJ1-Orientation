"""
    自定义transform方法实现三通道和直方图均衡化
    ImageLoader结合DataLoader进行批训练，每次抛出一批数据
"""
import torch
import torch.utils.data as Data
from torchvision import transforms
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
import random
import numpy as np
import os
import cv2

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
        # 产生直方图均衡化(w*h*channel)，transforms处理时默认最后一维才是通道数
        new_image = np.stack((temp, cv2.equalizeHist(temp80), cv2.equalizeHist(temp60)),axis=2)
  #      print(new_image.shape)
        return new_image


BATCH_SIZE = 8

transforms = { #存储transform的字典：键：数据集名称；值：对应数据集的transform
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


#dataset = ImageFolder(r'.\FinalData\LGE', transforms)
dataset = {  #同时读取LGE目录下的三个数据夹
    x: ImageFolder(os.path.join(r'.\FinalData\LGE', x), transforms[x]) #对train,validation,test图片变换字典的读取
    for x in ['train', 'validation', 'test']
}

loader = {
    x: Data.DataLoader(dataset=dataset[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    for x in ['train', 'validation', 'test']
}

"""
train_loader = loader['train'] #提取每个训练器，可以在循环中通过访问列表来代替
valid_loader = loader['validation']
test_loader = loader['test']
"""

def show_batch():
    for epoch in range(1):
        for setName in ['train', 'validation', 'test']:
            for step, (batch_x, batch_y) in enumerate(loader[setName]):
                #此时batch_x存在三通道3*w*h， 需要依次显示或者存储
                #print("steop:{}, batch_pic:{}, batch_label:{}".format(step, batch_x, batch_y))
                for i in range(BATCH_SIZE):
                    print("setName:{}, step:{}, batch_pic:{}, batch_label:{}".format(setName, step, i, batch_y[i]))
                    '''
                    plt.imshow(batch_x[i,0,:,:].squeeze()) #squeeze降维图片为2*2才能显示
                    plt.axis('off')
                    plt.show() #注释则不需要手动关闭程序即可继续
                    '''
               #     print(batch_x.shape) 批量*宽*通道*长，第三维才是通道
                    cv2.imshow("show", np.asarray(batch_x[i,0,:,:].squeeze())) #转换tensor为cv2可以读取的格式
             #       cv2.waitKey(0)  # 使用waitKey暂停在图片显示界面
                    cv2.imshow("show1", np.asarray(batch_x[i, 1, :, :].squeeze()))
              #      cv2.waitKey(1)  # 使用waitKey暂停在图片显示界面
                    cv2.imshow("show2", np.asarray(batch_x[i, 2, :, :].squeeze()))
                    cv2.waitKey(2)  # 使用waitKey暂停在图片显示界面



if __name__ == '__main__':
    show_batch()