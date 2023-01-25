"""
使用该py文件进行文件预处理，运行前需要手动在桌面对应位置data_itr建立三个nil文件夹
"""
import os
import nibabel as nib
import re
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
            newChildImg = childImg[:,:,i] #取出列表
            cv2.imwrite(extractFilename, newChildImg) #保存图片为png格式
    return

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


def finalData(relative_path_read, relative_path_save, extract_data_itr, final_data_itr):
    """
    对extract_data_itr文件夹下所有png图片进行八种朝向变换，然后存储到对应的final_data_itr子目录0~7下
    即生成了图片-朝向对(X_t,O_t)
    """
    imgPath = os.listdir(extract_data_itr) #返回图片组下的所有子图片路径列表
    for item in tqdm(imgPath): #用进度条表示读取进度
        childPath = os.path.join(relative_path_read, item)
        childImg = cv2.imread(childPath, cv2.IMREAD_GRAYSCALE) #读取单个灰度图片为ndarray(imread只能读取相对路径)
  #      cv2.imshow("show", childImg)
 #       cv2.waitKey(0) #使用waitKey暂停在图片显示界面
        orientTrans(childImg, item, relative_path_save) #对图片朝向进行变化，并存储到对应的文件夹中：使用相对路径\LGE\i\item
    return


if __name__=="__main__":
    data_itr = r"C:\Users\lenovo\Desktop\MyCode\InitData"
    extract_data_itr = r"C:\Users\lenovo\Desktop\MyCode\ExtractData"
    final_data_itr = r"C:\Users\lenovo\Desktop\MyCode\FinalData"
    dataType = ["C0","LGE","T2"]
    for itr in dataType:
    #依次遍历（三种不同模式图片）文件夹data_itr下所有nil文件解压为2D灰度图片，并存储到extract_data_itr下存储
        searchPath(os.path.join(extract_data_itr, itr)) #创建存放解压结果的目标文件夹
        extractData(os.path.join(data_itr, itr), os.path.join(extract_data_itr, itr))
        for i in range(8): #创建存放朝向变化后结果的目标文件夹
            searchPath(os.path.join(final_data_itr, itr, str(i)))
        finalData(os.path.join(r".\ExtractData", itr), os.path.join(r".\FinalData", itr),
                  os.path.join(extract_data_itr, itr), os.path.join(final_data_itr, itr))