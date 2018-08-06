from bin.Func import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 设置路径
path = 'D:/htxt/data/1.tiff'
img = read_tiff(path)
row = 1500
column = 1500
img = img[:row, :column]

# plt.imshow(img, cmap='gray')  # 显示图片
# plt.axis('off')  # 不显示坐标轴
# plt.show()

test_data = slide_window(img)
test_data = np.asanyarray(test_data, np.float32)
test_data = test_data[:, :, :, np.newaxis]
# print(test_data.shape)

del img

# 读取cnn模型

with tf.Session() as sess:
    data = test_data
    saver = tf.train.import_meta_graph('./model.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    # print(data.shape)

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    logits = graph.get_tensor_by_name("logits_eval:0")
    # 批处理
    prob = []
    y_ = np.ones((row*column))
    for batches, _ in minibatches(test_data, y_, 10000):
        batches_result = sess.run(logits, feed_dict={x:batches})
        prob.append(batches_result)

    prob = np.asanyarray(prob)
    prob.resize((prob.shape[0]*prob.shape[1], prob.shape[2]))


    result = tf.argmax(prob, 1).eval()


result.resize((row,column))
plt.imshow(result, cmap='gray')
plt.axis('off')  # 不显示坐标轴
plt.show()
# 保存图片
# cv.imwrite('./result.jpg', labels)
