"""
使用该py文件进行文件预处理，运行前需要手动在桌面对应位置data_itr建立三个nil文件夹
相较于原始的程序dataPrePros.py，加入了随机分割80%、10%、10%训练集、验证集、测试集部分
先分割再分成八类

缺陷：由于最后直接建立在当前目录子目录下（这样可以避免randomPartitionData与变更朝向的函数嵌套），
    除了0~7类外还有原始图片，导致不能直接使用ImageFolder读取，需要手动删除这部分图片
"""
import os
import nibabel as nib
import re
import random
import cv2
from tqdm import tqdm
import numpy as np

def searchPath(filename):
#寻找文件filename，如果不存在则在当前路径创建新文件
    if not os.path.isdir(filename):
        os.makedirs(filename)
    return

def extractData(data_itr, extract_data_itr):
    """
    遍历文件夹data_itr下所有nil文件解压为2D灰度图片，并存储到extract_data_itr下存储
    """
    preFilename = data_itr #主文件夹名称
    for item in os.listdir(data_itr): #子nil文件名称
        allFilename = os.path.join(preFilename,item) #拼接出完整Nil文件名称
        targChildFilename = re.sub(re.escape(".nii.gz") + '', '', item)  # 使用正则库去掉nil文件后缀nii.gz并加上_i后缀
        img = nib.load(allFilename) #加载nil数据的三维列表
        childImg = img.get_fdata() #提前存储，否则取切片会出错
        size = img.shape[2] #访问解压出的图片数目
        for i in range(size): #加载单个图片，并且存储到目标文件夹
            extractFilename = os.path.join(extract_data_itr, targChildFilename+"_"+str(i)+".png") #拼接出目标图片文件路径
            newChildImg = childImg[:,:,i] #取出单个图片的列表
            cv2.imwrite(extractFilename, newChildImg) #保存图片为png格式
    return

def randomPartitionData(relative_path_read, relative_path_save):
    """
    接受来自相对路径relative_path_read的图片，并随机分割为80%、10%、10%训练集、验证集、测试集后，
    存储到relative_path_save\setName文件夹中
    其中extract_data_itr和final_data_itr中存放的是绝对路径
    """
    imgPath = os.listdir(relative_path_read)  # 返回图片组下的所有子图片路径列表
    num = len(imgPath)
    train_num = int(num*0.8)
    valid_num = int(num*0.1)
    test_num = num-train_num-valid_num
    tempindice = list(range(num))
    random.shuffle(tempindice) #产生0~num-1的随机序列用作索引，以进行随机划分(没有返回值)
    indice = np.array(tempindice) #列表不可以使用列表作为索引，但是Array可以
    setNameDic = {"train":indice[0:train_num], "validation":indice[train_num:(train_num+valid_num)],
                  "test":indice[(train_num+valid_num):num]} #包头不包尾
    for name in setName:
        for item in tqdm(np.array(imgPath)[setNameDic[name]]): #列表不可以使用列表作为索引，但是Array可以
            #两层循环，第一层用于产生三种：训练集、验证集、测试集，第二层借助索引遍历子目录文件名称
            curImg = cv2.imread(os.path.join(relative_path_read, item), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(relative_path_save, name, item), curImg)


def orientTrans(img, img_name, file_path):
    """
    对单个图片img，依次进行八种变换，并存储到filepath\i\img_name中
    """
    orientDir = {3:cv2.ROTATE_180, 5:cv2.ROTATE_90_CLOCKWISE, 6:cv2.ROTATE_90_COUNTERCLOCKWISE} #记录旋转的朝向与i对应关系
    size = img.shape
    for i in range(8):
        if i in [1,2]:
            saveImg = cv2.flip(img, 2-i) #对1和2调用flip函数进行水平或垂直翻转
        elif i in [3, 5, 6]:
            saveImg = cv2.rotate(img, orientDir[i]) #对3、5、6调用rotate进行直角旋转
        elif i==4:
            new = np.zeros((size[1], size[0]), np.uint8) #建立二维空数组(size[1], size[0])
            for j in range(size[1]):
                for k in range(size[0]):
                    new[j][k] = img[k][j]
            saveImg = new
        elif i==7:
         #   new = [[0] * size[1] for i in range(size[0])]  # 建立二维空数组
            new = np.zeros((size[1], size[0]), np.uint8)
            for j in range(size[1]):
                for k in range(size[0]):
                    new[j][k] = img[size[0]-1-k][size[1]-1-j]
            saveImg = new
        else:
            saveImg = img

        cv2.imwrite(os.path.join(file_path, str(i), img_name), saveImg)  # 只能使用相对路径
    return

def finalData(relative_path, final_data_itr):
    """
    对relative_path(由于就在子目录下，只需要一个路径的相对和绝对)文件夹下所有png图片进行八种朝向变换，
    然后存储到对应的final_data_itr子目录0~7下，即生成了图片-朝向对(X_t,O_t)
    """
    imgPath = os.listdir(final_data_itr) #返回图片组下的所有子图片路径列表
    for i in range(8): #创建存放朝向变化后结果的目标文件夹
        searchPath(os.path.join(final_data_itr, str(i))) # (为了防止读出新建的0~7空文件夹，必须在此后建立)
    for item in tqdm(imgPath): #用进度条表示读取进度
        childPath = os.path.join(relative_path, item)
        childImg = cv2.imread(childPath, cv2.IMREAD_GRAYSCALE) #读取单个灰度图片为ndarray(imread只能读取相对路径)
  #      cv2.imshow("show", childImg)
 #       cv2.waitKey(0) #使用waitKey暂停在图片显示界面
        orientTrans(childImg, item, relative_path) #对图片朝向进行变化，并存储到对应的文件夹中：使用相对路径\LGE\i\item
    return


if __name__=="__main__":
    data_itr = r"C:\Users\lenovo\Desktop\MyCode\InitData"
    extract_data_itr = r"C:\Users\lenovo\Desktop\MyCode\ExtractData"
    final_data_itr = r"C:\Users\lenovo\Desktop\MyCode\FinalData"
    dataType = ["C0","LGE","T2"]
    setName = ["train", "validation", "test"]
    for itr in dataType:
        #依次遍历（三种不同模式图片）文件夹data_itr下所有nil文件解压为2D灰度图片，并存储到extract_data_itr下存储
        searchPath(os.path.join(extract_data_itr, itr)) #创建存放解压结果的目标文件夹
        extractData(os.path.join(data_itr, itr), os.path.join(extract_data_itr, itr)) #建立压缩后的文件夹
        #对解压后的文件夹随机划分出训练集、验证集、测试集，并存储到文件夹final_data_itr下
        for name in setName: #创建存放划分后的图集的文件夹
            searchPath(os.path.join(final_data_itr, itr, name))
        # 对三种模态的图片依次随即划分为80%、10%、10%
        randomPartitionData(os.path.join(r".\ExtractData", itr), os.path.join(r".\FinalData", itr))
        for name in setName: #在当前文件夹下，建立0~7种朝向变化的文件夹并进行变化
            finalData(os.path.join(r".\FinalData", itr, name), os.path.join(final_data_itr, itr, name))



