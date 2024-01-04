import os
import time

import cv2

image_files = os.listdir("./pixel_data/images/train")

new_data_dir = "./new_pixel_data/"

output_counter = 0
padding_size = 15

print(len(image_files))
for i in range(len(image_files)):
    image = cv2.imread("./pixel_data/images/train/" + image_files[i])

    try:
        file = open("./pixel_data/labels/train/" + image_files[i][:-4] + ".txt", "r").read()
    except FileNotFoundError:
        continue

    for x, line in enumerate(file.split("\n")):
        line_class = line[0]
        box = line[1:].split(" ")[1:]
        box = [float(x) for x in box]

        minx = int((box[0] - box[2] / 2) * image.shape[1])
        miny = int((box[1] - box[3] / 2) * image.shape[0])
        maxx = int((box[0] + box[2] / 2) * image.shape[1])
        maxy = int((box[1] + box[3] / 2) * image.shape[0])

        cropped = image.copy()[miny:maxy, minx:maxx]

        if not os.path.exists(new_data_dir + line_class):
            os.mkdir(new_data_dir + line_class)

        if not cv2.imwrite(new_data_dir+line_class+"/"+image_files[i]+str(output_counter)+".png", cropped):
            raise "EEAZZ"
        else:
            output_counter += 1
    print(i)