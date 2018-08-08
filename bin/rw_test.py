from PIL import Image
import os
imagePath = 'D:/htxt/Dataset/'
resultFile = 'D:/htxt/Output/'
counter = 0
for filename in os.listdir(imagePath):


    if filename[-4:] != 'tiff':
        continue

    counter += 1
    img = Image.open(imagePath+filename)
    results_file_jpg = "SAR目标提取_" + str(counter) + "_Results.jpg"
    img.save(resultFile + results_file_jpg)
    Image.register_save()