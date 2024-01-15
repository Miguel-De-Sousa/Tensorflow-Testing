import tensorflow
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data=keras.datasets.fashion_mnist
(trainimages, trainlabels), (testimages, testlabels) = data.load_data()
classnames=['T-shirt', 'Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Boot']

trainimages=trainimages/255.0
testimages=testimages/255.0

model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(128, activation="relu"),keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(trainimages, trainlabels, epochs=5)

testloss, testaccuracy = model.evaluate(testimages, testlabels)
print(testaccuracy)