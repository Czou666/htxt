from bin.Func import *
import numpy as np
import tensorflow as tf
from skimage import io
import sys
import os

# 获取.exe文件所在目录
exe_path = os.path.dirname(sys.argv[0])

# 载入插件
io.use_plugin('matplotlib', 'imread')
io.use_plugin('pil', 'imsave')

# 忽略警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 设置路径
try:
    imagePath = sys.argv[1] + '/'
    resultFile = sys.argv[2] + '/'
except:
    imagePath = 'D:/htxt/Dataset/'
    resultFile = "D:/htxt/Output/"

# 一张一张处理
for filename in os.listdir(imagePath):

    # 判断文件是否为.tiff格式
    if filename[-4:] != 'tiff':
        continue

    # 读取图片
    img = io.imread(imagePath + filename)
    row, column = img.shape
    # 显示图片
    # io.imshow(img, cmap='gray')
    # io.show()

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
        saver = tf.train.import_meta_graph(exe_path + '/' + 'model/model.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint(exe_path + '/' + 'model/'))
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
        cnn_result = tf.argmax(prob, 1).eval()

    # 将类别按图像原来形状排列，得到二值图
    cnn_result.resize((row, column))

    # 形态学处理
    labels = morphology_process(cnn_result, min_size=64, disk_value=2)

    # 转换为uint8类型，并*255
    result_image = np.zeros((row, column), dtype=np.uint8)
    result_image[labels == True] = 255
    # 显示结果
    # io.imshow(result_image, cmap='gray')
    # io.show()

    #############################保存文件############################

    # # 设置文件名
    results_file_bmp = "SAR目标提取_" + filename[:-5] + "_Results.bmp"
    results_file_xml = "SAR目标提取_" + filename[:-5] + "_Results.xml"

    # 保存图像
    io.imsave(resultFile + results_file_bmp, result_image)
    # # 保存XML文件
    xml_output(filename, results_file_bmp, resultFile + results_file_xml)



print('successfully detected!')
input('press enter key to exit') #这儿放一个等待输入是为了不让程序退出












