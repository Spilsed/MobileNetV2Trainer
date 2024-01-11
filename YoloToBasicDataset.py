import os
import cv2
import matplotlib.pyplot as plt
from keras.layers import RandomBrightness, RandomContrast, RandomRotation, RandomFlip, RandomTranslation, Rescaling
from keras.models import Sequential
from tqdm import tqdm
import random as r

image_files = os.listdir("./pixel_data/images/train")

new_data_dir = "./new_pixel_data/"

output_counter = 0
padding_size = 15

augmentations = 25
backgrounds = 5

data_augmentation = Sequential([
    RandomBrightness(0.5),
    RandomContrast(0.5),
    RandomTranslation(0.25, 0.25),
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.25),
])

if not os.path.exists(new_data_dir + "5"):
    os.mkdir(new_data_dir + "5")

def clampi(value, lower, upper):
    if value > upper:
        return int(upper)
    elif value < lower:
        return int(lower)
    else:
        return int(value)

for i in tqdm(range(len(image_files))):
    image = cv2.imread("./pixel_data/images/train/" + image_files[i])

    try:
        file = open("./pixel_data/labels/train/" + image_files[i][:-4] + ".txt", "r").read()
    except FileNotFoundError:
        continue

    for x, line in enumerate(file.split("\n")):
        line_class = line[0]
        box = line[1:].split(" ")[1:]
        box = [float(x) for x in box]

        minx = clampi((box[0] - box[2] / 2) * image.shape[1] - padding_size, 0, 1920)
        miny = clampi(((box[1] - box[3] / 2) * image.shape[0]) - padding_size, 0, 1080)
        maxx = clampi(((box[0] + box[2] / 2) * image.shape[1]) + padding_size, 0, 1920)
        maxy = clampi(((box[1] + box[3] / 2) * image.shape[0]) + padding_size, 0, 1080)

        cropped = image.copy()[miny:maxy, minx:maxx]

        if not os.path.exists(new_data_dir + line_class):
            os.mkdir(new_data_dir + line_class)

        # Actual images
        for aug in range(augmentations):
            auged = data_augmentation(cropped).numpy()
            cv2.imwrite(
                new_data_dir + line_class + "/" + image_files[i] + str(output_counter) + "_" + str(aug) + ".png", auged)

        # Background images
        for background in range(backgrounds):
            back = image.copy()
            center = (r.randrange(200, 1080 - 200), r.randrange(400, 1920 - 400))
            center = [int(i) for i in center]
            width = int(r.randrange(100, 400) / 2)
            height = int(r.randrange(100, 200) / 2)

            back = back[center[0] - height: center[0] + height, center[1] - width: center[1] + width]

            cv2.imwrite(new_data_dir + "5/" + image_files[i] + str(output_counter) + "_" + str(background) + ".png", back)

        output_counter += 1