from __future__ import print_function

import random
import sys

import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import (LSTM, Activation, Convolution1D, Dense, Dropout,
                          Flatten, Bidirectional)
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.preprocessing import sequence
from keras.utils.data_utils import get_file
from keras.utils import plot_model

filename = "songdata.csv"
artist = "bob dylan"
df = pd.read_csv(filename)
x_ds, y_ds = df['text'].tolist(), df['artist'].tolist()
print('Artist: ' + artist)
for i in range(len(y_ds)):
    y_ds[i] = y_ds[i].lower()
yy = [i for i,val in enumerate(y_ds) if artist in val]
text = ''
for i in yy:
    text += x_ds[i]

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
maxlen = 120
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(Dropout(0.25))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)
model.summary()
plot_model(model, to_file='model.png')


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# train the model, output generated text after each iteration
for iteration in range(1, 100):
    print()
    print('-' * 50)
    print('Iteration', iteration)

    # define the checkpoint
    filepath= artist.lower() + "_weights.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    
    model.fit(X, y,
              batch_size=128,
              epochs=1, callbacks=callbacks_list)

    start_index = random.randint(0, len(text) - maxlen - 1)
    with open('bobdylan.txt', 'a') as fp:
        fp.write('Iteration: ' + str(iteration))
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print()
            print('----- diversity:', diversity)
            fp.write('\n----- diversity:' + str(diversity) + '\n')
            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            # sentence=generated
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x = np.zeros((1, maxlen, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.

                preds = model.predict(x, verbose=0)[0]
                next_index = sample(preds, diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char
                fp.write(next_char)
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()
            fp.write('\n')
