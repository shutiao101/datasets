import tensorflow as tf
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt

x_data = datasets.load_iris().data  # 150*4 数据
y_data = datasets.load_iris().target  # 150*1 标签

np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

x_train = x_data[:-30]
y_train = y_data[:-30]
x_test = x_data[-30:]
y_test = y_data[-30:]

train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)  # 分成batch,目的是提高效率
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)  # 分成batch,目的是提高效率
'''
在这里卡了一万年, 对于w1 and b1 的数据类型和Variable要必须同时满足
'''
# 创建权重和偏置的随机张量
w1_random = tf.random.truncated_normal([4, 3], stddev=0.1, seed=1)
b1_random = tf.random.truncated_normal([3], stddev=0.1, seed=1)

# 将随机张量转换为tf.float64类型
w1 = tf.Variable(tf.cast(w1_random, dtype=tf.float64))
b1 = tf.Variable(tf.cast(b1_random, dtype=tf.float64))


Ir = 0.1
train_loss_results = []  # 均为画图所用,为列表
test_acc = []  # 均为画图所用,为列表
epoch = 500
loss_all = 0

for epoch in range(epoch):
    for step, (x_train, y_train) in enumerate(train_db):  # 迭代一次也是要把所有数据都遍历一遍, 从而完成梯度下降, 一共4steps
        with tf.GradientTape() as tape:
            y = tf.matmul(x_train, w1) + b1
            y = tf.nn.softmax(y)  # y and y_ are both matrices
            y_ = tf.one_hot(y_train, depth=3, dtype=tf.float64)  # y and y_ are both matrices
            loss = tf.reduce_mean(tf.square(y_ - y))  # error of mean square (均方误差),但是loss已经是一个值了 [shape is ()]
            loss_all += loss.numpy()  # .numpy()表示只取值,毕竟loss本质还是一个tensor
        grads = tape.gradient(loss, [w1, b1])  # grads is a matrix

        w1.assign_sub(Ir * grads[0])
        b1.assign_sub(Ir * grads[1])

    print(f'epoch: {epoch}, loss: {loss_all / 4}')
    train_loss_results.append(loss_all / 4)
    loss_all = 0  # 归零,为下一个epoch做准备

    #  测试部分,已经完成对一个epoch(即一次迭代)对w1和b1的更新
    total_correct, total_number = 0, 0  # 即完成了初始化,又保证下一次的初始化
    for x_test, y_test in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)  # y.shape is (30, 3)
        '''
        经过测试,pred = tf.cast(pred, dtype=y_test.dtype)这句话不加就是不行,尽管都是tf.int32但是就是不行
        '''
        pred = tf.argmax(y, axis=1)  # 找到概率最大的
        pred = tf.cast(pred, dtype=y_test.dtype)
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 若分类正确,correct则为1,否则correct为0, correct.shape is (30,)
        correct = tf.reduce_sum(correct)  # 经测试,185次迭代后,所有预测都符合测试集,即 correct==30 after 185 epochs
        total_correct += int(correct)  # 将correct 转换成int类型整数,否则无法直接除
        total_number += x_test.shape[0]  # 总的训练数据的计数
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("-------------------------")

plt.title('Loss Function Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss_results, label='$Loss$')
plt.legend()  # 显示图例
plt.show()

plt.title('Acc Curve')
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.plot(test_acc, label='$Accuracy$')
plt.legend()  # 显示图例
plt.show()
