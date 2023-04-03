import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import json
import random

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Load the intents file
with open('Intentss.json', 'r') as f:
    intents = json.load(f)

# Initialize stemmer
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


for intent in intents["intents"]:  # loop through each intent in the intents list
    tag = intent["intent"]  # get the intent name as the tag
    tags.append(tag)  # append the tag to the tags list
    context_in = intent["context"]["in"] if "context" in intent and "in" in intent[
        "context"] else ""  # get the context in, or an empty string if it doesn't exist
    context_out = intent["context"]["out"] if "context" in intent and "out" in intent[
        "context"] else ""  # get the context out, or an empty string if it doesn't exist
    clear_context = intent["context"]["clear"] if "context" in intent and "clear" in intent[
        "context"] else False  # get the clear context flag, or False if it doesn't exist

    extension_func = intent["extension"]["function"] if "extension" in intent and "function" in intent[
        "extension"] else ""  # get the extension function, or an empty string if it doesn't exist
    extension_entities = intent["extension"]["entities"] if "extension" in intent and "entities" in intent[
        "extension"] else False  # get the extension entities flag, or False if it doesn't exist
    extension_responses = intent["extension"]["responses"] if "extension" in intent and "responses" in intent[
        "extension"] else []  # get the extension responses list, or an empty list if it doesn't exist


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
# Define function to process user input
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]
    bag = bag_of_words(stemmed_tokens, all_words)
    return bag

# Define function to predict intent
def predict_intent(text):
    bag = preprocess(text)
    prediction = model.predict(np.array([bag]))
    predicted_index = np.argmax(prediction)
    tag = tags[predicted_index]
    confidence = prediction[0][predicted_index]
    if confidence > 0.6:
        for intent in intents['intents']:
            if intent['intent'] == tag:
                response = random.choice(intent['responses'])
                return response
    else:
        return "I'm sorry, I didn't understand that. Can you please try again?"

# Start the chatbot
print("GeniSys: Hi, how can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    else:
        response = predict_intent(user_input)
        print("GeniSys:", response)
