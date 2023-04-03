import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import string
import json
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Embedding, Flatten, Input
from tensorflow.keras.models import Model
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer
from nltk.corpus import stopwords
import re
import nltk
nltk.download('punkt')



with open('intent.json', 'r') as f:
    intent = json.load(f)

stemmer = PorterStemmer()
def tokenize(sentence):

    return nltk.word_tokenize(sentence)
def stem(word):

    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):

    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag

all_words = []  # initialize empty list to hold all words
tags = []  # initialize empty list to hold all tags
xy = []  # initialize empty list to hold tuples of (words, tag, context_in, context_out, clear_context, extension_func, extension_entities, extension_responses)


for intent in intent["intents"]:  # loop through each intent in the intents list
    tag = intent["intent"]  # get the intent name as the tag
    tags.append(tag)  # append the tag to the tags list

    for pattern in intent["text"]:  # loop through each pattern in the intent text list
        w = tokenize(pattern)  # tokenize the pattern into a list of words
        all_words.extend(w)  # add the words to the all_words list
        xy.append((w, tag))  # append a tuple of (words, tag, context_in, context_out, clear_context, extension_func, extension_entities, extension_responses) to the xy list


# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")  # print the number of patterns in the xy list
print(len(tags), "tags:", tags)  # print the number of tags and the tags list
print(len(all_words), "unique stemmed words:", all_words)  # print the number of unique stemmed words and the all_words list

# Set the training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)
# convert X_train and y_train to numpy arrays



X_train = np.array(X_train)
y_train = np.array(y_train)

print(X_train)
print(y_train)
print(X_train.shape)
print(y_train.shape)
# Set the hyperparameter
num_epochs = 150
batch_size = 8
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 7
output_size = len(tags)


# Define the model architecture
input_layer = Input(shape=(input_size,))
embedding_layer = Embedding(input_dim=input_size, output_dim=32)(input_layer)
flatten_layer = Flatten()(embedding_layer)
hidden_layer1 = Dense(32, activation='selu')(flatten_layer)
dropout_layer1 = Dropout(0.1)(hidden_layer1)
hidden_layer2 = Dense(16, activation='selu')(dropout_layer1)
dropout_layer2 = Dropout(0.1)(hidden_layer2)
output_layer = Dense(output_size, activation='softmax')(dropout_layer2)

model = Model(inputs=input_layer, outputs=output_layer)



# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# Train the model
# Train the model
model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)


# Save the model
model.save('my_model.h5')
