from xml.etree import ElementTree as ET
import time
def xml_output(filename, results_file_jpg, results_file_xml):

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
    second_node = ET.SubElement(first_node_5, 'ResultsFile')
    second_node.text = results_file_jpg

    # 生成树
    tree = ET.ElementTree(root)
    # 保存
    tree.write(results_file_xml, encoding='UTF-8', xml_declaration=True)






