from bin.Func import *
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os



# 忽略警告
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# 设置路径
imagePath = 'D:/htxt/Dataset/'
resultFile = "D:/htxt/Output/"

# 一张一张处理
counter = 0
for filename in os.listdir(imagePath):

    # 判断文件是否为.tiff格式
    if filename[-4:] != 'tiff':
        continue

    # 读取图片
    img = read_tiff(imagePath+filename)
    row, column = img.shape

    # 滑窗
    test_data = slide_window(img)
    test_data = np.asanyarray(test_data, np.float32)
    test_data = test_data[:, :, :, np.newaxis]

    # 删除无用变量
    del img

    # 读取cnn模型并分类
    with tf.Session() as sess:
        data = test_data
        # print(data.shape)

        # 读取模型
        saver = tf.train.import_meta_graph('model/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('model/'))
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        logits = graph.get_tensor_by_name("logits_eval:0")

        # 批处理
        prob = []
        y_ = np.ones((row * column))
        for batches, _ in minibatches(test_data, y_, 10000):
            batches_result = sess.run(logits, feed_dict={x: batches})
            prob.append(batches_result)

        # 得到所有像素点类别的概率
        prob = np.asanyarray(prob)
        prob.resize((prob.shape[0] * prob.shape[1], prob.shape[2]))

        # 得到所有像素点的类别
        result = tf.argmax(prob, 1).eval()

    # 将类别按图像原来形状排列，得到二值图
    result.resize((row, column))

    # 显示二值图
    plt.imshow(result, cmap='gray')
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    #############################保存文件############################
    counter += 1
    # 设置文件名
    results_file_jpg = "SAR目标提取_" + str(counter) + "_Results.jpg"
    results_file_xml = "SAR目标提取_" + str(counter) + "_Results.xml"
    # 矩阵转图像
    result_image = matrix_to_image(result)
    # 保存图像
    result_image.save(resultFile + results_file_jpg)
    # 保存XML文件
    xml_output(filename, results_file_jpg, resultFile + results_file_xml)












