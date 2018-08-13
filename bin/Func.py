from skimage import morphology
import numpy as np
from xml.etree import ElementTree as ET
import time

##########################读数据##########################

def crop_image(img, start_row, end_row, start_column, end_column):
    region = img[start_row:end_row, start_column:end_column]
    return region

def get_regions(img, pos_set):
    '''
    得到建筑物类和非建筑物类区域样本数据
    :param img: 采集样本的图像
    :param pos_set: 图像中建筑 or 非建筑区域的坐标集合
                    格式为( (start_row, end_row, start_column, end_column),
                            (start_row, end_row, start_column, end_column),
                            .....
                            (start_row, end_row, start_column, end_column)
                            )
                    建筑物和非建筑物应分别存储
    :return: 采集区域的数据
    '''
    regions = []
    i = 0
    for pos in pos_set:

        if len(pos) != 4:
            print("位置填写错误")
            return None
        # print(pos)
        regions.append(crop_image(img, pos[0], pos[1], pos[2], pos[3]))
        # 显示图像
        # io.imshow(regions[i], cmap='gray')  # 显示图片
        # io.show()
        i += 1
    return regions

def slide_window(img, window_size=8, data_type='test'):
    '''
    滑窗程序，返回滑窗后的图像块，存储于一个list当中
    :param img: 输入图像
    :param window_size: 窗的大小，默认为8
    :param data_type: 属于训练图像还是测试图像：'test'为测试图像，'train'为训练图像
    :return: 返回一个list，保存滑窗后的图像块
    '''

    img_row, img_column = img.shape
    slice_set = []
    # 如果为测试数据，先扩充图像
    if data_type == "test":
        add_row = img[-window_size:]
        img = np.row_stack((img, add_row))
        add_column = img[:, -window_size:]
        img = np.column_stack((img, add_column))
    if data_type == "train":
        img_row = img_row - window_size
        img_column = img_column - window_size

    for i in range(img_row):
        for j in range(img_column):
            img_slice = img[i:i+window_size, j:j+window_size]
            slice_set.append(img_slice)

    return slice_set

def get_training_set(regions, region_type):
    '''
    得到建筑物/非建筑物的训练样本集和类标签
    :param regions: 建筑物区域/非建筑物区域
    :param region_type: 'building'代表输入的区域为建筑物，'nonbuilding'代表非建筑物
    :return: data为滑窗后的结果，存在一个list中；
             labels为类标签，建筑物为1，非建筑物为0，存在一个list中
    '''

    data = []
    labels = []
    for region in regions:
        data = data + slide_window(region, data_type='train')


    if region_type == 'bulding':
        labels = [1] * data.__len__()
    if region_type == 'nonbuilding':
        labels = [0] * data.__len__()
    return data, labels

############################ CNN #########################################

# 定义批处理函数
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):

    '''

    :param inputs: 样本
    :param targets: 样本的标签
    :param batch_size: 一次处理多少图像块
    :param shuffle:是否随机抽取
    :return: 返回分批的数据
    '''

    # assert condition:如果condition为假，则报错AssertionError
    assert len(inputs) == len(targets)
    if shuffle:
        # 得到样本位置索引
        indices = np.arange(len(inputs))
        # 随机排序
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        # yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面的值。
        # 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码开始执行。
        yield inputs[excerpt], targets[excerpt]

###########################形态学处理################################

def morphology_process(cnn_result, min_size=64, disk_value=1):

    row, column = cnn_result.shape
    labels = np.zeros((row, column), dtype=np.bool)
    labels[cnn_result == 1] = True

    # 去除散点及小区域
    morphology.remove_small_objects(labels, min_size=min_size, connectivity=1, in_place=True)
    # io.imshow(labels)
    # io.show()
    # 图像闭运算
    labels = morphology.closing(labels, morphology.disk(disk_value))
    # io.imshow(labels)
    # io.show()
    return labels

###########################输出################################

# 输出XML文件
def xml_output(filename, results_file_name, path):
    '''

    :param filename: 当前输入图像名称
    :param results_file_name: 输出图片名称
    :param path: 输出的xml文件名称
    :return:
    '''

    # 第一层
    root = ET.Element('Research', {'ImageName': filename, 'Direction': '高分软件大赛'})

    # 第二层
    first_node_1 = ET.SubElement(root, 'Department')
    first_node_1.text = '中国民航大学'
    first_node_2 = ET.SubElement(root, 'Date')
    first_node_2.text = time.strftime("%Y-%m-%d")
    first_node_3 = ET.SubElement(root, 'PluginName')
    first_node_3.text = '目标提取'
    first_node_4 = ET.SubElement(root, 'PluginClass')
    first_node_4.text = '提取'
    first_node_5 = ET.SubElement(root, 'Results', {'Coordinate': 'Pixel'})

    # 第三层
    second_node = ET.SubElement(first_node_5, 'ResultsFile')
    second_node.text = results_file_name

    # 生成树
    tree = ET.ElementTree(root)
    # 保存 （输出目录+输出文件名，UTF-8编码方式，声明中显示编码方式）
    tree.write(path, encoding='UTF-8', xml_declaration=True)
