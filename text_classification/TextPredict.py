import tensorflow as tf
from tensorflow import keras
import numpy as np
import json



# with open('word_index.json', 'w') as f:
# 	json.dump(word_index, f)

# with open('reverse_word_index.json', 'w') as f:
# 	json.dump(reverse_word_index, f)


with open('word_index.json') as f:
	word_index = json.load(f)

with open('reverse_word_index.json') as f:
	reverse_word_index = json.load(f)


def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])


def review_encode(s):
	encoded = [1]

	for word in s:
		if word in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded


#######


model = keras.models.load_model(".\model.h5")

with open(".\\sample\\badReviewTest.txt", encoding="utf-8") as f:
	for line in f.readlines():

		# Removeing all symbols
		nline = line.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"", "").strip().split(" ")

		# encoding 
		encode = review_encode(nline)
		encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
		predict = model.predict(encode)

		print("encoding = ")
		print(encode)
		print("\nOringal text = \n" + line + "\n")
		#print(encode)
		print("predict = \n", predict)
		print("Result = ", predict[0])

		if (predict[0][0] > 0.5):
			print("The model thinks your review is a GOOD review")
		else:
			print("The model thinks your review is a BAD review")



