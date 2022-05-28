# =====================================================================================================
# PROBLEM C4 
#
# Build and train a classifier for the sarcasm dataset. 
# The classifier should have a final layer with 1 neuron activated by sigmoid.
# 
# Do not use lambda layers in your model.
# 
# Dataset used in this problem is built by Rishabh Misra (https://rishabhmisra.github.io/publications).
#
# Desired accuracy and validation_accuracy > 75%
# =======================================================================================================

import json
import tensorflow as tf
import numpy as np
import urllib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split


nltk.download('stopwords')

def solution_C4():
    data_url = 'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/sarcasm.json'
    urllib.request.urlretrieve(data_url, 'sarcasm.json')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_size = 20000

    sentences = []
    labels = []
    # YOUR CODE HERE
    with open('sarcasm.json','r') as f:
        data=json.load(f)

    for item in data:
        sentences.append(item['headline'])
        labels.append(item['is_sarcastic'])

    
    sentences = np.array(sentences)
    labels = np.array(labels)

    train = sentences[:training_size]
    test = sentences[training_size:]
    train_label = labels[:training_size]
    test_label = labels[training_size:]


    tokenizer = Tokenizer(num_words = vocab_size,
                            oov_token=oov_tok,)
    tokenizer.fit_on_texts(sentences)

    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(train)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type,padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(test)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length,truncating=trunc_type,padding=padding_type)
    
    model = tf.keras.Sequential([
    # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32,return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16)),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.fit(padded, train_label, epochs=10,batch_size=64, validation_data=(testing_padded,test_label))    
    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_C4()
    model.save("model_C4.h5")