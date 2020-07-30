import tensorflow as tf
from tensorflow import keras
import numpy as np


data = keras.datasets.imdb

#(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

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
#train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
#test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])

#####

def review_encode(s):
	encoded = [1]

	for word in s:
		if word in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

model = keras.models.load_model("..\model.h5")

with open("badReviewTest.txt", encoding="utf-8") as f:
	for line in f.readlines():

		# Removeing all symbols
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")

		# encoding 
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
		predict = model.predict(encode)
		print(line)
		print(encode)
		print(predict[0])

		if (predict[0][0] > 0.5):
			print("The model thinks your review is a GOOD review")
		else:
			print("The model thinks yoru review is a BAD review")



