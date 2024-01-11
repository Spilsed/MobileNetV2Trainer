import tensorflow as tf
from keras.layers import Dense, GlobalAveragePooling1D
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory

BATCH_SIZE = 1
IMG_SIZE = (200, 200)

train_dataset = image_dataset_from_directory("./new_pixel_data",
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             label_mode="categorical",
                                             image_size=IMG_SIZE)

model = MobileNetV2(input_shape=(200, 200, 3), classes=5, weights=None)
model.summary()

for layer in model.layers[:-23]:
    layer.trainable = False

model.summary()

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=10)
model.save("./MobileNetV2")
