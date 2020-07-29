# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


data = keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = data.load_data()



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



# Since the color of the image is not important, the data can be modify to black and white
train_images = train_images/255.0
test_images = test_images/255.0


# print(train_labels[0])
# print(train_images[7])

# Using matplot to plot the image 
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()



## Creating a model

# softmax function makes it such that all the output add up to 1
# This is done to minmic probability.

model =keras.Sequential([
	keras.layers.Flatten(input_shape=(28,28)),
	keras.layers.Dense(128, activation="relu"),
	keras.layers.Dense(10, activation="softmax")
	])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# epochs determined the number of the same images show to the model
# The order of the input might influence the parameter of the network.
# This is done to increase the accracy of the model.
model.fit(train_images, train_labels, epochs=10)



test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Acc: ", test_acc)