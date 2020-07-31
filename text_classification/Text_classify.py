import tensorflow as tf
from tensorflow import keras
import numpy as np


data = keras.datasets.imdb

# only take the most frequence 10k words
# The words are represented in integer
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

#print(train_data[0])

# getting the mapping for the words
word_index = data.get_word_index()

# shift the index by 3 for each word
# this is done to compromise data that are not in the right length
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# DATA PREPROCESS
## Trimming the data
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])


#print(decode_review(test_data[0]))




# MODEL

model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))     # configure output neuron to become sigmodial between 0, 1, indicating the movie is good or bad.



model.summary()

# adam
# loss function= "binary_crossentrop" this is select for binary output
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])


# spliting training data into 2 set

x_val = train_data[:10000]
x_train = train_data[10000:]

# valadation data
y_val = train_labels[:10000]
y_train = train_labels[10000:]

# the batch_size 
fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val,y_val), verbose=1)

results = model.evaluate(test_data, test_labels)


# showing train result
print(results)

test_review = test_data[0]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[0]))
print("Actual: " + str(test_labels[0]))
print(results)


model.save("model.h5")

