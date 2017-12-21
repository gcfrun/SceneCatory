# coding:utf-8
import tensorflow as tf
import numpy as np
import data_input

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) # 变量的初始值为截断正太分布
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 80],name='x')
x_image = tf.reshape(x, [-1,8,10,1]) #将输入按照 conv2d中input的格式来reshape，reshape

"""
# 第一层
"""
W_conv1 = weight_variable([3, 3, 1, 12])
b_conv1 = bias_variable([12])
h_conv1 = tf.nn.elu(conv2d(x_image, W_conv1) + b_conv1)

"""
# 第二层
"""
W_conv2 = weight_variable([3, 3, 12, 24])
b_conv2 = bias_variable([24])
h_conv2 = tf.nn.elu(conv2d(h_conv1, W_conv2) + b_conv2)

# 第三层 是个全连接层,输入维数16*5*64, 输出维数为1024
W_fc1 = weight_variable([8 * 10 * 24, 1024])
b_fc1 = bias_variable([1024])
h_conv2_flat = tf.reshape(h_conv2, [-1, 8*10*24])
h_fc1 = tf.nn.elu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)
keep_prob = tf.placeholder(tf.float32,name='keep_prob') # 这里使用了drop out,即随机安排一些cell输出值为0，可以防止过拟合
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第四层，输入1024维，输出3维，也就是具体的0~2分类
W_fc2 = weight_variable([1024, 3])
b_fc2 = bias_variable([3])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2) # 使用softmax作为多分类激活函数
y_ = tf.placeholder(tf.float32, [None, 3],name='y_')

#y_conv+1e-10 特别重要
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv+1e-10), reduction_indices=[1]))# 损失函数，交叉熵


tf.summary.scalar("loss", cross_entropy)

train_step = tf.train.AdamOptimizer(10e-6).minimize(cross_entropy) # 使用adam优化  过大会造成梯度爆炸
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1)) # 计算准确度
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name='accuracy')

tf.summary.scalar('Accuracy', accuracy)

merged_summary = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(tf.global_variables_initializer()) # 变量初始化
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
for i in range(8500):
    batch = data_input.InputClass().next_batch(50)
    if i%50 == 0:
        print('data:',batch[0].shape,'label:',batch[1].shape)
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_: batch[1], keep_prob: 1.0})

        result = sess.run(merged_summary,
                          feed_dict={
                              x: batch[0], y_: batch[1], keep_prob: 1.0})
        writer.add_summary(result, i)

        print("step %d, training accuracy %g"%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
saver.save(sess, 'save/model.ckpt')
testData = data_input.InputClass().test_data()
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: testData[0], y_: testData[1], keep_prob: 1.0}))