import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.optimizers import Adam
from keras.utils import image_dataset_from_directory

print(tf.config.list_physical_devices())

BATCH_SIZE = 256
IMG_SIZE = (200, 200)

train_dataset = image_dataset_from_directory("./new_pixel_data",
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             label_mode="categorical",
                                             image_size=IMG_SIZE)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)

model = MobileNetV2(input_shape=(200, 200, 3), classes=6, weights=None)

# for layer in model.layers[:-23]:
#     layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

while input("Train again? (y/n)") != "n":
    model.fit(train_dataset, epochs=5)

model.save("./MobileNetV2")
