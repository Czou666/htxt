from bin.Func import *
import numpy as np
from sklearn import model_selection
import tensorflow as tf

########################## 得到训练集 ##########################

# 设置路径
path = 'D:/htxt/data/1.tiff'
img = read_tiff(path)
# 设置样本坐标
buliding_pos = ((1029, 1098, 402, 471),
                (875, 914, 31, 70),
                (501, 530, 6, 35),
                (568, 607, 933, 972),
                (970, 1019, 711, 760),
                (849, 913, 885, 949)
                    )
nonbuliding_pos = ((1135, 1204, 1011, 1080),
                   (755, 809,1122, 1176),
                   (338, 397,1218, 1277),
                   (53, 128,277, 352),
                   (610, 669,115, 174),
                   (1147, 1191,127, 171),
                   (241, 295, 164, 218)
                   )

# 第十幅图样本



building_regions = get_regions(img, buliding_pos)
nonbuilding_regions = get_regions(img, nonbuliding_pos)

# 得到滑窗后的样本集
building_data, building_labels = get_training_set(building_regions, 'bulding')
nonbuilding_data, nonbuilding_labels = get_training_set(nonbuilding_regions, 'nonbuilding')
data = building_data + nonbuilding_data
labels = building_labels + nonbuilding_labels

# 转换成tensorflow中训练网络需要的数据格式
# 输入的数据类型为ndarray data中的数值为float32 labels中的数值为int32
data = np.asanyarray(data, np.float32)
data = data[:, :, :, np.newaxis]
labels = np.asarray(labels, np.int32)

# 划分测试和训练数据
x_train, x_test, y_train, y_test = model_selection.train_test_split(data, labels,
                                test_size=0.25, random_state=0, stratify=labels)


# 打印数据维度
# print(x_train.shape)
# print(y_train.shape)

# 删除无用变量
del building_data
del nonbuilding_data
del building_labels
del nonbuilding_labels
del data
del labels
del buliding_pos
del nonbuliding_pos
del img
del path
del building_regions
del nonbuilding_regions


########################## 构建网络 ##########################

x = tf.placeholder(tf.float32, shape=[None, 8, 8, 1], name='x')
y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

# 第一个卷积层（8——>3)
# conv1格式:(?, 6, 6 32)
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[3, 3],
    padding="valid",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# 第二个卷积层(3->1)
# conv2格式:(?, 2. 2, 64)
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[2, 2],
    padding="valid",
    activation=tf.nn.relu,
    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
re1 = tf.reshape(pool2, [-1, 1 * 1 * 64])

# 全连接层
# dense1格式：(?, 256)
dense1 = tf.layers.dense(inputs=re1,
                         units=256,
                         activation=tf.nn.relu,
                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.002))
# dense2格式：(?, 2)
logits = tf.layers.dense(inputs=dense1,
                         units=2,
                         activation=None,
                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                         kernel_regularizer=tf.contrib.layers.l2_regularizer(0.002))


# 计算损失函数 softmax loss
loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
# 选择优化器，最小化损失函数
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
# 得到类别与真实值的对比向量：若该样本的预测类别与实际类别相同则为1，否则为0
correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
# 计算准确率
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

b = tf.constant(value=1,dtype=tf.float32)
logits_eval = tf.multiply(logits,b,name='logits_eval')
########################## 初始化 ##########################


n_epoch = 150
batch_size = 32
saver=tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

########################## 训练 ##########################
for epoch in range(n_epoch):

    # training
    train_loss, train_acc, n_batch = 0, 0, 0
    for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
        _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
        train_loss += err;
        train_acc += ac;
        n_batch += 1
    print("   train loss: %f" % (train_loss / n_batch))
    print("   train acc: %f" % (train_acc / n_batch))

    # validation
    val_loss, val_acc, n_batch = 0, 0, 0
    for x_test_a, y_test_a in minibatches(x_test, y_test, batch_size, shuffle=False):
        err, ac = sess.run([loss, acc], feed_dict={x: x_test_a, y_: y_test_a })
        val_loss += err;
        val_acc += ac;
        n_batch += 1
    print("   validation loss: %f" % (val_loss / n_batch))
    print("   validation acc: %f" % (val_acc / n_batch))


########################## 保存模型 ##########################
model_path = './model.ckpt'
saver.save(sess,model_path)
sess.close()

'''
   train loss: 0.060419
   train acc: 0.976618
   validation loss: 0.379473
   validation acc: 0.911078
'''