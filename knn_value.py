# coding:utf-8

import numpy as np
import data_input
import operator
import gabor_gist

"""
data:测试数据
dataset:训练数据源
labels:训练数据标签
k:k邻近
返回测试数据标签
"""
def knn(data,dataset,labels,k):
  #计算行数
  dataSetSize = dataset.shape[0]
  #矩阵作差（每行一个样本，行里的每个元素都是属性）
  diffMat = np.tile(data, (dataSetSize, 1)) - dataset
  #差平方
  sqDiffMat = diffMat ** 2
  #差平方和（列元素相加）
  sqDistance = sqDiffMat.sum(axis=1)
  #开根号（欧式距离）
  distance = sqDistance ** 0.5
  #从小到大排序（元素下标）
  sortedDistIndicies = distance.argsort()
  #投票
  classCount = {}
  for i in range(k):
      #获取对应的标签
      voteIlabel = labels[sortedDistIndicies[i]]
      #票数+1
      classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
  #投票从大到小排序
  sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

if __name__ == '__main__':
    #读取数据
    features, img_labels, pca_array = data_input.getTrainResult()
    test_images,test_labels = data_input.getTestSample()
    #求每个测试数据的标签
    scale = [5, 8, 11, 14]
    test_knn_labels = []
    for i in range(len(test_images)):
        print ('测试第'+str(i+1)+'个样本...')
        #转灰度
        gray_img = test_images[i].convert('L')
        #gist特征
        gist = gabor_gist.getGist(gray_img,4,8,scale)
        #降维
        pca_gist = np.dot(np.array(gist), np.transpose(features))
        #knn求真实标签,k=3
        label = knn(pca_gist,pca_array,img_labels,3)
        test_knn_labels.append(label)
    #显示正确率
    num = 0
    for i in range(len(test_labels)):
        if (test_knn_labels[i] == test_labels[i]):
            num += 1
    percent = float(num)/float(len(test_labels))*100
    print('正确个数:'+str(num))
    print('总个数:' + str(len(test_labels)))
    print('正确率：'+str(percent)+'')
