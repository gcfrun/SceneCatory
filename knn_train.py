# coding:utf-8

import numpy as np
import gabor_gist
import data_input

"""
data_array:特征向量数组
k:降低到k维
返回k个特征向量
"""
def pca(data_array,k):
    #计算每个纬度的平均值
    mean=np.array([np.mean(data_array[:,i]) for i in range(data_array.shape[1])])
    #数据中心化
    norm_X=data_array-mean
    #散度矩阵
    scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
    #获得数组(特征值，特征向量）
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    #降序排列特征值索引
    val_index = np.argsort(-eig_val)
    #选择前k个特征向量
    features=np.array([eig_vec[i] for i in val_index[:k]])
    return features

if __name__ == '__main__':
    #获取训练数据
    img_array, img_labels = data_input.getTrainSample()
    #求gist特征
    gist_array = []
    scale = [5, 8, 11, 14]
    for i in range(len(img_array)):
        gray_img = img_array[i].convert('L')
        gist = gabor_gist.getGist(gray_img,4,8,scale)
        gist_array.append(gist)
        print('训练第'+str(i+1)+'个样本...')
    features = pca(np.array(gist_array),80)
    #求80维下的数据
    pca_array = np.dot(np.array(gist_array), np.transpose(features))
    #存储数据
    np.save('TrainResult/pca_array.npy', pca_array)
    np.save('TrainResult/features.npy', features)
    np.save('TrainResult/img_labels.npy', img_labels)