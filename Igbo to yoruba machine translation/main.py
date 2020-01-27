import keras
import tensorflow as tf
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from numpy import argmax
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu


from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import pandas as pd
import io
import csv


from keras.preprocessing.text import Tokenizer


def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    X = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    X = pad_sequences(X, maxlen=length, padding='post')
    return X


def encode_output(sequence, vocab_size):
    encoded = []
    encoded = to_categorical(sequence, num_classes=vocab_size)
    y = np.array(encoded)
    y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    return y


# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)


def main(x):
    """
    Takes in an input X and returns an output y
    Where X is the English domain language.
    X: (input) has to be a list of the English domain Sentence.
    Y is the corresponding IGBO domain language.
    """

    eng_tokenizer = create_tokenizer(x)
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    eng_length = max_length(x)

    trainx = encode_sequences(eng_tokenizer, eng_length, x)

    model = load_model('model.h5')
    prediction = model.predict(trainx)


    return prediction



if __name__ == '__main__':
    x = ['i got Home']
    value = main(x)
    print(value)
