import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

import tensorflow as tf
import cv2
import os
import numpy as np

labels = ['A', 'B']
img_size=200
def get_data(data_dir):
    data=[]
    for label in labels:
        path= os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1]
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype=object)

train = get_data('D:\Paradigms\Python\OpenCV\Input\Train')
val = get_data('D:\Paradigms\Python\OpenCV\Input\Test')

x_train = []
y_train=[]
x_val=[]
y_val=[]

for feature, label in train:
    
    x_train.append(feature)
    y_train.append(label)
for feature, label in val:
    x_val.append(feature)
    y_val.append(label)

x_train = np.array(x_train)/255.0
x_val = np.array(x_val)/255.0

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

datageneration=ImageDataGenerator(
                                featurewise_center=False,
                                samplewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_std_normalization=False,
                                zca_whitening=False,
                                rotation_range=30,
                                zoom_range=0.2,
                                width_shift_range=0.1,
                                height_shift_range=0.1,
                                horizontal_flip=True,
                                vertical_flip=False
                                )

datageneration.fit(x_train)

BSLmodel=Sequential()
BSLmodel.add(Conv2D(32,3,padding="same", activation="relu",input_shape=(200,200,3)))
BSLmodel.add(MaxPool2D())
BSLmodel.add(Conv2D(32, 3, padding="same", activation="relu"))
BSLmodel.add(MaxPool2D())
BSLmodel.add(Conv2D(64, 3, padding="same", activation="relu"))
BSLmodel.add(MaxPool2D())
BSLmodel.add(Dropout(0.4))
BSLmodel.add(Flatten())
BSLmodel.add(Dense(128, activation="relu"))
BSLmodel.add(Dense(2, activation="softmax"))
BSLmodel.summary()

optimisation=Adam(learning_rate=0.001)
BSLmodel.compile(optimizer= optimisation, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history=BSLmodel.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

accuracy=history.history['accuracy']
val_acc = history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range=range(100)

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, accuracy, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

model_json = BSLmodel.to_json()
with open("BSLmodel.json","w") as json_file:
    json_file.write(model_json)
BSLmodel.save_weights("model.h5")
print("Saved model to disk")