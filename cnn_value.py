# coding:utf-8

import tensorflow as tf
import data_input
import numpy as np


# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
#     return tf.Variable(initial)
#
# def bias_variable(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)
#
# def conv2d(x, W):
#     """
#     tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
#     前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
#     input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
#     filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
#     strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
#     padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
#     use_cudnn_on_gpu 是否使用cudnn加速。默认是True
#     """
#     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
#
#
# sess = tf.InteractiveSession()
#
# x = tf.placeholder(tf.float32, [None, 80])
# x_image = tf.reshape(x, [-1,8,10,1]) #将输入按照 conv2d中input的格式来reshape，reshape
#
# """
# # 第一层
# # 卷积核(filter)的尺寸是3*3, 通道数为1，输出通道为32，即feature map 数目为32
# # 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*16*5*32
# # 9*9，共有32个通道,共有?个批次
# # 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*16*15*32
# """
# W_conv1 = weight_variable([3, 3, 1, 12])  # 卷积是在每个5*5的patch中算出32个特征，分别是patch大小，输入通道数目，输出通道数目
# b_conv1 = bias_variable([12])
# h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)
#
# """
# # 第二层
# # 卷积核3*3，输入通道为32，输出通道为64。
# # 卷积前图像的尺寸为 ?*16*5*32， 卷积后为?*16*5*64
# # 池化后，输出的图像尺寸为?*16*5*64
# """
# W_conv2 = weight_variable([3, 3, 12, 24])
# b_conv2 = bias_variable([24])
# h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2) + b_conv2)
#
# # 第三层 是个全连接层,输入维数16*5*64, 输出维数为1024
# W_fc1 = weight_variable([8 * 10 * 24, 1024])
# b_fc1 = bias_variable([1024])
# h_conv2_flat = tf.reshape(h_conv2, [-1, 8*10*24])
# h_fc1 = tf.nn.elu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
# keep_prob = tf.placeholder(tf.float32) # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
# h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
#
# # 第四层，输入1024维，输出3维，也就是具体的0~2分类
# W_fc2 = weight_variable([1024, 3])
# b_fc2 = bias_variable([3])
# y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2,name='prob') # 使用softmax作为多分类激活函数
# y_ = tf.placeholder(tf.float32, [None, 3])
#
# #y_conv+1e-10 特别重要
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv+1e-10), reduction_indices=[1]))# 损失函数，交叉熵
#
#
# train_step = tf.train.AdamOptimizer(10e-6).minimize(cross_entropy) # 使用adam优化  过大会造成梯度爆炸
# correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#
#
#
# meta_path = 'save/model.ckpt.meta'
# model_path = 'save/model.ckpt'
# saver = tf.train.Saver()
#
# sess.run(tf.global_variables_initializer())
# saver.restore(sess, model_path)  # 导入变量值
# print("test accuracy %g"%accuracy.eval(feed_dict={
#     x: test_data()[0], y_: test_data()[1], keep_prob: 1.0}))





sess = tf.InteractiveSession()
saver = tf.train.import_meta_graph('save/model.ckpt.meta')

graph = tf.get_default_graph()
x = graph.get_tensor_by_name('x:0')

y_ = graph.get_tensor_by_name('y_:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')

accuracy = graph.get_tensor_by_name('accuracy:0')

saver.restore(sess, 'save/model.ckpt')

testData = data_input.InputClass().test_data()
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testData[0], y_: testData[1], keep_prob: 1.0}))
