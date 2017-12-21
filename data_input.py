# coding:utf-8

"""
总共三个场景，每个场景有80个训练数据，20个测试数据
训练时随机从每个场景中取50个数据
测试时随机从每个场景中取10个数据
ACB-1,AnFpark-2,PDFpark-3
"""

import numpy as np
import random
import gabor_gist
import pylab as pl
from PIL import Image

def getTrainSample():
    # 获得每个场景的要取的图片
    ACB_index = []
    AnFpark_index = []
    PDFpark_index = []
    for i in range(80):
        ACB_index.append(i)
        AnFpark_index.append(i)
        PDFpark_index.append(i)
    random.shuffle(ACB_index)
    random.shuffle(AnFpark_index)
    random.shuffle(PDFpark_index)

    # 读取图片
    img_array = []
    img_labels = []
    for i in range(50):
        image = Image.open('TrainData/ACB/' + 'ACB_' + str(ACB_index[i])+'.jpeg')
        img_array.append(image)
        img_labels.append(1)
        image = Image.open('TrainData/AnFpark/' + 'AnFpark_' + str(AnFpark_index[i])+'.jpeg')
        img_array.append(image)
        img_labels.append(2)
        image = Image.open('TrainData/FDFpark/' + 'FDFpark_' + str(PDFpark_index[i])+'.jpeg')
        img_array.append(image)
        img_labels.append(3)

    return img_array,img_labels


def getTestSample():
    # 获得每个场景的要取的图片
    ACB_index = []
    AnFpark_index = []
    PDFpark_index = []
    for i in range(80,100):
        ACB_index.append(i)
        AnFpark_index.append(i)
        PDFpark_index.append(i)
    random.shuffle(ACB_index)
    random.shuffle(AnFpark_index)
    random.shuffle(PDFpark_index)

    # 读取图片
    img_array = []
    img_labels = []
    for i in range(10):
        image = Image.open('TestData/ACB/' + 'ACB_' + str(ACB_index[i])+'.jpeg')
        img_array.append(image)
        img_labels.append(1)
        image = Image.open('TestData/AnFpark/' + 'AnFpark_' + str(AnFpark_index[i])+'.jpeg')
        img_array.append(image)
        img_labels.append(2)
        image = Image.open('TestData/FDFpark/' + 'FDFpark_' + str(PDFpark_index[i])+'.jpeg')
        img_array.append(image)
        img_labels.append(3)

    return img_array,img_labels


def getTrainResult():
    features = np.load('TrainResult/features.npy')
    img_labels = np.load('TrainResult/img_labels.npy')
    pca_array = np.load('TrainResult/pca_array.npy')
    return features,img_labels,pca_array

class InputClass:
    features, img_labels, pca_array = getTrainResult()
    train_images,train_labels = getTrainSample()
    test_images,test_labels = getTestSample()

    temp_data = pca_array
    temp_label = img_labels
    '''
    返回的图片必须是二维数组，每一行是一个80维的gist特征，一次返回num个gist特征
    返回的标签也必须是一个二维数组，每一行是一个三维的标签，除了对应的真实值为1，剩余两个都是0，一次返回num个标签组
    '''
    def next_batch(self,num):
        r_data = []
        r_label = []

        # 获得每个场景的要取的图片
        data_index = []
        if len(self.temp_data)<num:
            self.temp_data = self.pca_array
            self.temp_label = self.img_labels
        for i in range(len(self.temp_data)):
            data_index.append(i)
        random.shuffle(data_index)
        for i in range(num):
            r_data.append(self.temp_data[data_index[i]])
            label = [0, 0, 0]
            label[self.temp_label[data_index[i]] - 1] = 1
            r_label.append(label)
        next_data = []
        next_label = []
        for i in range(num,len(self.temp_data)):
            next_data.append(self.temp_data[data_index[i]])
            next_label.append(self.temp_label[data_index[i]])
        self.temp_data = np.array(next_data)
        self.temp_label = np.array(next_label)
        return np.array(r_data), np.array(r_label)

    def test_data(self):
        # 求每个测试数据的标签
        scale = [5, 8, 11, 14]
        test_gists = []
        test_label_arr = []
        for i in range(len(self.test_images)):
            # 转灰度
            gray_img = self.test_images[i].convert('L')
            # gist特征
            gist = gabor_gist.getGist(gray_img, 4, 8, scale)
            # 降维
            pca_gist = np.dot(np.array(gist), np.transpose(self.features))
            test_gists.append(pca_gist)

            label = [0, 0, 0]
            label[self.test_labels[i] - 1] = 1
            test_label_arr.append(label)
        return np.array(test_gists),np.array(test_label_arr)
