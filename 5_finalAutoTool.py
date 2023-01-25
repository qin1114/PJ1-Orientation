# -*- coding: utf-8 -*-

"""
本文件由ui通过pyuic5 -o finalAutoTool1.py test1AutoTool.ui直接导出
对导出的类添加初始化方法setupUi和导入包并show，然后定义主函数
"""


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

import sys
import os
import traceback
import cv2
from PIL import Image
import numpy as np
import all_rc #加载图片资源的all_rc.py文件

import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch import nn


def searchClass(path):
    #传入路径名称，查找其中包含的字符从而获得图片类别：["LGE", "T2", "C0"]
    if path.find("LGE")!=-1:
        return "LGE"
    elif path.find("T2")!=-1:
        return "T2"
    elif path.find("C0")!=-1:
        return "C0"

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


def predictSigImg(imgPath):
    #对传入的单张图片进行朝向预测，返回数据标签对的二元元组(img, imgLabel)
    #数据预处理
    img0 = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE) #图片以灰度图的格式读取
    img = Image.fromarray(img0) #变为PIL格式
    transform = transforms.Compose([ #定义预测数据变换方式(同训练时)
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        AddChannel(), #暂时不使用外来传入参数
        transforms.ToTensor()
    ])
    img_new = transform(img).reshape([1,3,256,256]) #进行了数据升维1*3*256*256
    #加载本地网络参数，先判断当前图片类型(目前只支持LGE格式)
    # typeNet = searchClass(imgPath)
    typeNet = "LGE"
    net=torch.load(r".\Results\LGE\saveNetAll_" + typeNet + ".tar", map_location=torch.device('cpu'))
    imgLabel = net(img_new).argmax().item() #获取类别
    return (img0, imgLabel)


def orientInverse(img, i):
    """
    对朝向为i的单个图片img，进行逆变换
    """
    orientDir = {3:cv2.ROTATE_180, 5:cv2.ROTATE_90_COUNTERCLOCKWISE, 6:cv2.ROTATE_90_CLOCKWISE} #记录旋转的朝向与i对应关系
    size = img.shape
    if i in [1, 2]:
        saveImg = cv2.flip(img, 2 - i)  # 对1和2调用flip函数进行水平或垂直翻转
    elif i in [3, 5, 6]:
        saveImg = cv2.rotate(img, orientDir[i])  # 对3、5、6调用rotate进行直角旋转
    elif i == 4:
        new = np.zeros((size[1], size[0]), np.uint8)  # 建立二维空数组(size[1], size[0])
        for j in range(size[1]):
            for k in range(size[0]):
                new[j][k] = img[k][j]
        saveImg = new
    elif i == 7:
        new = np.zeros((size[1], size[0]), np.uint8)
        for j in range(size[1]):
            for k in range(size[0]):
                new[j][k] = img[size[0] - 1 - k][size[1] - 1 - j]
        saveImg = new
    else:
        saveImg = img

    #print(saveImg)
      #  cv2.imwrite(os.path.join(file_path, str(i), img_name), saveImg)  # 只能使用相对路径
    return saveImg


class Ui_Mywidget(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.MainWindow = QtWidgets.QWidget() #建立主窗口，并作为参数传入setupUi方法
        self.curIndex = 0  # 定义全局变量，表示当前的显示和处理的图片序号
        self.setupUi(self.MainWindow)
        self.filename = [] #用于存储列表名称

    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(707, 410)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(Form.sizePolicy().hasHeightForWidth())
        Form.setSizePolicy(sizePolicy)
        Form.setLayoutDirection(QtCore.Qt.LeftToRight)
        Form.setAutoFillBackground(False)
        Form.setStyleSheet("background:rgb(255, 255, 255)")
        self.gridLayout_2 = QtWidgets.QGridLayout(Form)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_14 = QtWidgets.QLabel(Form)
        self.label_14.setText("")
        self.label_14.setObjectName("label_14")
        self.verticalLayout.addWidget(self.label_14)
        self.OpenButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.OpenButton.sizePolicy().hasHeightForWidth())
        self.OpenButton.setSizePolicy(sizePolicy)
        self.OpenButton.setStyleSheet("background:rgb(211, 211, 211)")
        self.OpenButton.setObjectName("OpenButton")

        self.OpenButton.clicked.connect(self.loadImg)  # 链接图片获取的方法loadImg

        self.verticalLayout.addWidget(self.OpenButton)
        self.AdjustButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AdjustButton.sizePolicy().hasHeightForWidth())
        self.AdjustButton.setSizePolicy(sizePolicy)

        self.AdjustButton.clicked.connect(self.adjustImg)  # 链接图片朝向纠正的方法 adjustImg

        self.AdjustButton.setStyleSheet("background:rgb(211, 211, 211)")
        self.AdjustButton.setObjectName("AdjustButton")
        self.verticalLayout.addWidget(self.AdjustButton)
        self.PredictButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PredictButton.sizePolicy().hasHeightForWidth())
        self.PredictButton.setSizePolicy(sizePolicy)
        self.PredictButton.setStyleSheet("background:rgb(211, 211, 211)")
        self.PredictButton.setObjectName("PredictButton")
        self.verticalLayout.addWidget(self.PredictButton)

        self.PredictButton.clicked.connect(self.predictImg)  # 链接批量预测并打印文本的方法predictImg

        self.SaveInitButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SaveInitButton.sizePolicy().hasHeightForWidth())
        self.SaveInitButton.setSizePolicy(sizePolicy)
        self.SaveInitButton.setStyleSheet("background:rgb(211, 211, 211)")

        self.SaveInitButton.clicked.connect(self.saveTxt)  # 链接存储预测文本的方法saveTxt

        self.SaveInitButton.setObjectName("SaveInitButton")
        self.verticalLayout.addWidget(self.SaveInitButton)
        self.SveChangedButton = QtWidgets.QPushButton(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.SveChangedButton.sizePolicy().hasHeightForWidth())
        self.SveChangedButton.setSizePolicy(sizePolicy)
        self.SveChangedButton.setStyleSheet("background:rgb(211, 211, 211)")
        self.SveChangedButton.setObjectName("SveChangedButton")

        self.SveChangedButton.clicked.connect(self.saveImg)  # 链接保存修正朝向后图片的方法saveImg

        self.verticalLayout.addWidget(self.SveChangedButton)
        self.label_15 = QtWidgets.QLabel(Form)
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.verticalLayout.addWidget(self.label_15)
        self.verticalLayout.setStretch(0, 2)
        self.verticalLayout.setStretch(1, 1)
        self.verticalLayout.setStretch(2, 1)
        self.verticalLayout.setStretch(3, 1)
        self.verticalLayout.setStretch(4, 1)
        self.verticalLayout.setStretch(5, 1)
        self.verticalLayout.setStretch(6, 5)
        self.horizontalLayout_3.addLayout(self.verticalLayout)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_12 = QtWidgets.QLabel(Form)
        self.label_12.setText("")
        self.label_12.setObjectName("label_12")
        self.horizontalLayout_2.addWidget(self.label_12)
        self.FileNameLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(7)
        sizePolicy.setHeightForWidth(self.FileNameLabel.sizePolicy().hasHeightForWidth())
        self.FileNameLabel.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.FileNameLabel.setPalette(palette)
        self.FileNameLabel.setStyleSheet("")
        self.FileNameLabel.setTextFormat(QtCore.Qt.MarkdownText)
        self.FileNameLabel.setObjectName("FileNameLabel")
        self.horizontalLayout_2.addWidget(self.FileNameLabel)
        self.label_13 = QtWidgets.QLabel(Form)
        self.label_13.setText("")
        self.label_13.setObjectName("label_13")
        self.horizontalLayout_2.addWidget(self.label_13)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setEnabled(True)

        self.pushButton.clicked.connect(self.leftImg)  # 链接加载上一张图片的方法leftImg

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton.sizePolicy().hasHeightForWidth())
        self.pushButton.setSizePolicy(sizePolicy)
        self.pushButton.setStyleSheet("border-image:url(:left.png)")
        self.pushButton.setText("")
        self.pushButton.setAutoDefault(False)
        self.pushButton.setDefault(False)
        self.pushButton.setFlat(False)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.label = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)

        self.label.setText("No.Slice:  0/ 0") #为了可以修正，从retranslateUi方法中移除

        self.pushButton_2 = QtWidgets.QPushButton(Form)

        self.pushButton_2.clicked.connect(self.rightImg)  # 链接切换到下一张图片的方法rightImg

        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_2.sizePolicy().hasHeightForWidth())
        self.pushButton_2.setSizePolicy(sizePolicy)
        self.pushButton_2.setStyleSheet("border-image:url(:right.png)")
        self.pushButton_2.setText("")
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.ClassLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.ClassLabel.sizePolicy().hasHeightForWidth())
        self.ClassLabel.setSizePolicy(sizePolicy)

        self.ClassLabel.setText("   Class: None ") #修正参数

        self.ClassLabel.setObjectName("ClassLabel")
        self.horizontalLayout.addWidget(self.ClassLabel)
        self.label_11 = QtWidgets.QLabel(Form)
        self.label_11.setText("")
        self.label_11.setObjectName("label_11")
        self.horizontalLayout.addWidget(self.label_11)
        self.horizontalLayout.setStretch(2, 1)
        self.horizontalLayout.setStretch(4, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.label_3 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setStyleSheet("border-image: url(:cover.png);")
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.verticalLayout_2.addWidget(self.label_3)
        self.tableWidget = QtWidgets.QTableWidget(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setObjectName("tableWidget")

        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)  # 列宽自动分配

        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.tableWidget.horizontalHeader().setDefaultSectionSize(95)
        self.verticalLayout_2.addWidget(self.tableWidget)
        self.verticalLayout_2.setStretch(1, 2)
        self.verticalLayout_2.setStretch(2, 16)
        self.verticalLayout_2.setStretch(3, 14)
        self.horizontalLayout_3.addLayout(self.verticalLayout_2)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_17 = QtWidgets.QLabel(Form)
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.verticalLayout_3.addWidget(self.label_17)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label_7 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_7.setPalette(palette)
        self.label_7.setObjectName("label_7")
        self.gridLayout.addWidget(self.label_7, 3, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_6.setPalette(palette)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 3, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_4.setPalette(palette)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(Form)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_8.setPalette(palette)
        self.label_8.setStyleSheet("font: 8pt \"Adobe Devanagari\";")
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 0, 2, 1)
        self.label_5 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_5.setPalette(palette)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)
        self.label_9 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(120, 120, 120))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.label_9.setPalette(palette)
        self.label_9.setStyleSheet("font: 8pt \"Adobe Devanagari\";")
        self.label_9.setObjectName("label_9")
        self.gridLayout.addWidget(self.label_9, 1, 2, 2, 1)
        self.AdaptedImageLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AdaptedImageLabel.sizePolicy().hasHeightForWidth())

        self.AdaptedImageLabel.setScaledContents(True)  # 让图片自适应label大小

        self.AdaptedImageLabel.setSizePolicy(sizePolicy)
        self.AdaptedImageLabel.setStyleSheet("border-image:url(:cover.png)")
        self.AdaptedImageLabel.setText("")
        self.AdaptedImageLabel.setObjectName("AdaptedImageLabel")
        self.gridLayout.addWidget(self.AdaptedImageLabel, 1, 1, 2, 1)
        self.verticalLayout_3.addLayout(self.gridLayout)
        self.label_10 = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setStyleSheet("border-image:url(:list.png)")
        self.label_10.setText("")
        self.label_10.setObjectName("label_10")
        self.verticalLayout_3.addWidget(self.label_10)
        self.label_16 = QtWidgets.QLabel(Form)
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.verticalLayout_3.addWidget(self.label_16)
        self.verticalLayout_3.setStretch(0, 1)
        self.verticalLayout_3.setStretch(1, 6)
        self.verticalLayout_3.setStretch(2, 5)
        self.verticalLayout_3.setStretch(3, 2)
        self.horizontalLayout_3.addLayout(self.verticalLayout_3)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 7)
        self.horizontalLayout_3.setStretch(2, 3)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 0, 0, 1, 1)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

        Form.show()  # 显示主窗口

    def loadImg(self):
        """
        单击Open按钮后，将图片显示在主窗口，并加载出文件名
        """
        try:
            #获取图片，需要注意的是，如果同时显示多个图片(使用复数getOpenFileNames)将会先显示第0张，
            # 并将所有的图片的路径暂存入列表filename中
            # 'Image files(*.jpg *.gif *.png)'
            self.filename, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "", os.getcwd(), #加载压缩文件
                                        "Image files(*.jpg *.gif *.png)")
            #self.label_3.setStyleSheet(None) #取消原先的图片背景
            self.label_3.setPixmap(QPixmap(""))  # 移除label上的图片
            self.label_3.setPixmap(QPixmap(self.filename[self.curIndex])) #将图片显示在主窗口上
            self.label_3.setScaledContents(True)  # 让图片自适应label大小

            _, finalname = os.path.split(self.filename[self.curIndex]) #提取文件名称
            self.FileNameLabel.setText(finalname) #文本标签显示图片名称

            size = len(self.filename)
            self.label.setText("No.Slice:  " + str(self.curIndex+1) + "/ " + str(size))
            self.ClassLabel.setText("   Class: "+str(searchClass(self.filename[self.curIndex])))

        except Exception as e:  # 异常时进行报错
            if str(e) == "list index out of range":  # 说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def rightImg(self):
        """
        单击Right按钮后，切换至下一张图片，并加载出文件名
        """
        try:
            size = len(self.filename)
            self.curIndex = (self.curIndex+1) % size #设置循环
            self.label_3.setPixmap(QPixmap(self.filename[self.curIndex]))  # 将图片显示在主窗口上
            self.label_3.setScaledContents(True)  # 让图片自适应label大小

            _, finalname = os.path.split(self.filename[self.curIndex])  # 提取文件名称
            self.FileNameLabel.setText(finalname)  # 文本标签显示图片名称

            size = len(self.filename)
            self.label.setText("No.Slice:  " + str(self.curIndex+1) + "/ " + str(size))
            self.ClassLabel.setText("   Class: " + str(searchClass(self.filename[self.curIndex])))

        except Exception as e:  # 异常时进行报错
            if str(e)=="integer division or modulo by zero": #说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def leftImg(self):
        """
        单击Left按钮后，切换至下一张图片，并加载出文件名
        """
        try:
            size = len(self.filename)
            self.curIndex = (self.curIndex-1+size) % size #设置循环
            self.label_3.setPixmap(QPixmap(self.filename[self.curIndex]))  # 将图片显示在主窗口上
            self.label_3.setScaledContents(True)  # 让图片自适应label大小

            _, finalname = os.path.split(self.filename[self.curIndex])  # 提取文件名称
            self.FileNameLabel.setText(finalname)  # 文本标签显示图片名称

            size = len(self.filename)
            self.label.setText("No.Slice:  " + str(self.curIndex+1) + "/ " + str(size))
            self.ClassLabel.setText("   Class: " + str(searchClass(self.filename[self.curIndex])))

        except Exception as e:  # 异常时进行报错
            if str(e) == "integer division or modulo by zero":  # 说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def adjustImg(self):
        #单击Adjust对当前图片进行朝向更正，并显示在右部的画布上
        try: #异常处理，要求必须先open
            (img, imgLabel) = predictSigImg(self.filename[self.curIndex]) #调用预测函数对单张图片朝向进行预测
            imgAdjusted = orientInverse(img, imgLabel) #调用朝向纠正函数纠正朝向
            # print(imgAdjusted.shape)
            # cv2.imshow("show", np.asarray(imgAdjusted))
            # 默认模式下暂时保存图片为png格式，建立个临时路径
            cv2.imwrite("temp.png", imgAdjusted)
            self.AdaptedImageLabel.setPixmap(QPixmap("temp.png"))  # 显示图片(不支持绝对路径)
            os.remove("temp.png")  # 删除该临时图片
        except Exception as e:  # 异常时进行报错
            if str(e) == "list index out of range":  # 说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def predictImg(self):
        #同时预测这些批量照片的结果，并在下方文本框中显示
        try:
            # cur_row = self.tableWidget.rowCount() #获取表格行数
            self.tableWidget.setRowCount(len(self.filename))  # 设置表格总行数
            cur_row = 0 #初始化行序号
            for item in self.filename:
                img0, label = predictSigImg(item) #挨个图片进行预测，返回预测类型
                _, finalname = os.path.split(item)  # 提取文件名称
                itemMap = [str(img0.shape[0]), str(img0.shape[1]), finalname, str(label)]
                for i in range(4):
                    item1 = QtWidgets.QTableWidgetItem(itemMap[i]) #新建item，逐个单元格写入，如果传入int型会显示为空！！
                    item1.setTextAlignment(Qt.AlignHCenter | Qt.AlignVCenter) #居中显示
                    # print(type(item1)) #判断是否成功创建item，可能返回空
                    self.tableWidget.setItem(cur_row, i, item1)#表格单元格的行标和列标都是从 0 开始
                cur_row += 1

        except Exception as e:  # 异常时进行报错
            if str(e) == "list index out of range":  # 说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def saveImg(self):
        #保存纠正后的图片按钮
        savename, _ = QFileDialog.getSaveFileName(self, 'save file', './', "Image files(*.png)")
        savename = savename.replace("/","\\")
        # print(savename)
        try: #异常处理，要求必须先open
            (img, imgLabel) = predictSigImg(self.filename[self.curIndex]) #调用预测函数对单张图片朝向进行预测
            imgAdjusted = orientInverse(img, imgLabel) #调用朝向纠正函数纠正朝向
            # print(imgAdjusted.shape)
            # cv2.imshow("show", np.asarray(imgAdjusted))
            cv2.imwrite(savename, imgAdjusted)
        except Exception as e:  # 异常时进行报错
            if str(e) == "list index out of range":  # 说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def saveTxt(self):
        #保存批量预测文本
        savename, _ = QFileDialog.getSaveFileName(self, 'save file', './', "txt(*.txt)")
        savename = savename.replace("/","\\")
        print(savename)
        try:
            cur_row = 0  # 初始化行序号
            for item in self.filename:
                img0, label = predictSigImg(item)  # 挨个图片进行预测，返回预测类型
                _, finalname = os.path.split(item)  # 提取文件名称
                tempLine = "Width: {}, Height: {}, FileName: {}, Orientation:{}\n".format(str(img0.shape[0]),
                                                                                          str(img0.shape[1]), finalname, str(label))
                if cur_row == 0:
                    writeMode = "w"  # "w"表示覆盖读写
                else:
                    writeMode = "a"  # "a"表示不覆盖续写
                with open(savename, writeMode) as f:  # 写入列表每一行
                    f.writelines(tempLine)
                cur_row += 1

        except Exception as e:  # 异常时进行报错
            if str(e) == "list index out of range":  # 说明还没有open图片
                pass
            else:
                print(str(Exception) + "\n")
                print(str(e))

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "AutoAdjust Tool"))
        self.OpenButton.setText(_translate("Form", "Open"))
        self.AdjustButton.setText(_translate("Form", "Adjust"))
        self.PredictButton.setText(_translate("Form", "Predict"))
        self.SaveInitButton.setText(_translate("Form", "Save(Predict.txt)")) #保存Predict后的表格为文本文件
        self.SveChangedButton.setText(_translate("Form", "Save(Adjust.png)")) #保存Adjust后的图片
        self.FileNameLabel.setText(_translate("Form", "File Name"))
  #      self.label.setText(_translate("Form", "No.Slice:  "+str(self.curIndice)+"/ "+str(size)))
   #     self.ClassLabel.setText(_translate("Form", "   Class: C0 "))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "Width"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "Height"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("Form", "FileName"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("Form", "Orientation"))
        self.label_7.setText(_translate("Form", "3"))
        self.label_6.setText(_translate("Form", "4"))
        self.label_4.setText(_translate("Form", "1"))
        self.label_8.setText(_translate("Form", "R/A"))
        self.label_5.setText(_translate("Form", "2"))
        self.label_9.setText(_translate("Form", "L/P"))


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Mywidget()
    # ui.setupUi(MainWindow)
    # MainWindow.show()
    sys.exit(app.exec_())
