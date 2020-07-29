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


print(train_labels[0])
print(train_images[7])

# Using matplot to plot the image 
plt.imshow(train_images[7], cmap=plt.cm.binary)
plt.show()