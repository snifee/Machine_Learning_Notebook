# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd

import nltk
from nltk.corpus import stopwords


def solution_B4():
    bbc = pd.read_csv('https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type='post'
    padding_type='post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    category = pd.get_dummies(bbc.category)

    df = pd.concat([bbc,category],axis=1)
    df = df.drop(columns='category')


    category = df[['tech', 'business', 'sport', 'entertainment', 'politics']].values
    article = df['text'].values

    stop_words = set(stopwords.words('english'))

    new_articles =[]

    for text in article:
      new_text = ""
      for w in text.split():
        if not w in stop_words:
            new_text = new_text + w + ' '
      new_articles.append(new_text)

    article = np.array(new_articles)

    train_text,test_text,train_label,test_label = train_test_split(article, category, train_size=training_portion, shuffle=False)

    tokenizer = Tokenizer(num_words = vocab_size,
                            oov_token=oov_tok)
    tokenizer.fit_on_texts(article)

    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(train_text)
    padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type,padding=padding_type)

    testing_sequences = tokenizer.texts_to_sequences(test_text)
    testing_padded = pad_sequences(testing_sequences,maxlen=max_length,padding=padding_type)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    cb = tf.keras.callbacks.EarlyStopping(patience=2)

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    model.fit(padded, train_label, epochs=25,batch_size=64, validation_data=(testing_padded,test_label))

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    model = solution_B4()
    model.save("model_B4.h5")
