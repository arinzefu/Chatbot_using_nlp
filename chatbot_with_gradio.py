import tensorflow as tf
import numpy as np
import nltk
from nltk.stem import PorterStemmer
import json
import random
import gradio as gr

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Load the intents file
with open('intent.json', 'r') as f:
    intents = json.load(f)

# Initialize stemmer
stemmer = PorterStemmer()

# Get all the words and tags
all_words = []  # initialize empty list to hold all words
tags = []  # initialize empty list to hold all tags
xy = []  # initialize empty list to hold tuples of (words, tag)

for intent in intents["intents"]:  # loop through each intent in the intents list
    tag = intent["intent"]  # get the intent name as the tag
    tags.append(tag)  # append the tag to the tags list

    for pattern in intent["text"]:  # loop through each pattern in the intent text list
        w = nltk.word_tokenize(pattern)  # tokenize the pattern into a list of words
        all_words.extend(w)  # add the words to the all_words list
        xy.append((w, tag))  # append a tuple of (words, tag) to the xy list

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stemmer.stem(w.lower()) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Define function to process user input
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token.lower()) for token in tokens]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for i, w in enumerate(all_words):
        if w in stemmed_tokens:
            bag[i] = 1
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

# Define the input and output interfaces
input_text = gr.inputs.Textbox(lines=2, label="Enter your message here")
output_text = gr.outputs.Textbox(label="Response")

# Create the Gradio interface
gradio_interface = gr.Interface(fn=predict_intent, inputs=input_text, outputs=output_text)

# Launch the interface
gradio_interface.launch()