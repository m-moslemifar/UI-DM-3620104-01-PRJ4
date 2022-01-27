from __future__ import unicode_literals

import string
import numpy as np
import pandas as pd
from hazm import *
from gensim.models import Word2Vec
import codecs
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Bidirectional
import matplotlib.pyplot as plt
import tensorflow_addons as tfa

t = pd.read_csv('train - small.csv', delimiter='\t', index_col=0)
v = pd.read_csv('dev - small.csv', delimiter='\t', index_col=0)
train = t[['comment', 'label_id']].copy()
validate = v[['comment', 'label_id']].copy()
frames = [train, validate]
data = pd.concat(frames, ignore_index=True)

data['comment'] = data['comment'].apply(lambda x: ' '.join(x.lower() for x in x.split()))

nmz = Normalizer()
stops = sorted(
    list(set([nmz.normalize(w) for w in codecs.open('stopwords.dat', encoding='utf-8').read().split('\n') if w]))
)

data['comment'] = data['comment'].apply(lambda x: ' '.join(x for x in x.split() if x not in stops))

lemmatizer = Lemmatizer()
data['comment'] = data['comment'].apply(lambda x: ' '.join([lemmatizer.lemmatize(x) for x in x.split()]))


class SentenceIterator:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for review in self.dataset.iloc[:, 0]:
            for sentence in review.split('.')[:-1]:
                words = [w for w in sentence.split(' ') if w != '']
                yield words


sentences = SentenceIterator(data)
w2v_model = Word2Vec(sentences=sentences)
w2v_model.train(sentences, epochs=10, total_examples=len(list(sentences)))
w2v_weights = w2v_model.wv.vectors
vocab_size, embedding_size = w2v_weights.shape


def word2token(word):
    try:
        return w2v_model.wv.key_to_index[word]
    # If word is not in index return 0. I realize this means that this
    # is the same as the word of index 0 (i.e. most frequent word), but 0s
    # will be padded later anyway by the embedding layer (which also
    # seems dirty, but I couldn't find a better solution right now)
    except KeyError:
        return 0


MAX_SEQUENCE_LENGTH = max([len(s) for s in list(sentences)])


class SequenceIterator:
    def __init__(self, dataset, seq_length):
        self.dataset = dataset

        self.translator = str.maketrans('', '', string.punctuation + '-')
        self.sentiments, self.ccount = np.unique(dataset.label_id, return_counts=True)

        self.seq_length = seq_length

    def __iter__(self):
        for comment, label_id in zip(self.dataset.iloc[:, 0], self.dataset.iloc[:, 1]):
            words = np.array([word2token(w) for w in comment.split(' ')[:self.seq_length] if w != ''])

            yield words, label_id


sequences = SequenceIterator(data, MAX_SEQUENCE_LENGTH)

set_x = []
set_y = []
for w, l in sequences:
    set_x.append(w)
    set_y.append(l)

set_x = pad_sequences(set_x, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
set_y = np.array(set_y)

x = set_x[:6000]
y = set_y[:6000]
val_x = set_x[6000:]
val_y = set_y[6000:]

model = Sequential()
model.add(Embedding(input_dim=vocab_size,
                    output_dim=embedding_size,
                    weights=[w2v_weights],
                    input_length=MAX_SEQUENCE_LENGTH,
                    mask_zero=True,
                    trainable=False))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

history = model.fit(x, y, epochs=5, batch_size=32,
                    validation_data=(val_x, val_y), verbose=1)

plt.figure(figsize=(12, 12))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.figure(figsize=(12, 12))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Accuracy')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

train_predictions = model.predict(x).tolist()
validate_predictions = model.predict(val_x).tolist()
train_standard = []
for i in train['label_id'].tolist():
    train_standard.append([i])
validate_standard = []
for i in validate['label_id'].tolist():
    validate_standard.append([i])

train_metric = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
train_standard = np.array(train_standard[:6000], np.int64)
train_predictions = np.array(train_predictions[:6000], np.float64)
train_metric.update_state(train_standard, train_predictions)
train_result = train_metric.result()
print('F1 score of train set predictions: ' + str(train_result.numpy()))

validate_metric = tfa.metrics.F1Score(num_classes=1, threshold=0.5)
validate_standard = np.array(validate_standard[:666], np.int64)
validate_predictions = np.array(validate_predictions[:666], np.float64)
validate_metric.update_state(validate_standard, validate_predictions)
validate_result = validate_metric.result()
print('F1 score of validation set predictions: ' + str(validate_result.numpy()))
