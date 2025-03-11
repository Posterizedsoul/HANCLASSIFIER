import pickle

import matplotlib.pyplot as plt
import nltk
import numpy as np
import tensorflow as tf

# KERAS
from keras.layers import (
    GRU,
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    Input,
)
from keras.models import Model
from keras.optimizers import Adam

# NLTK
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# SKLEARN
from tensorflow.keras.layers import Layer

from constants import *


class helpers:
    def __init__(self):
        self.__unwanted_cols = [
            "Authorization",
            "Accountability",
            "Authentication",
            "Non_Repudiation",
            "Availability",
            "Confidentiality",
            "Other",
        ]

        self.pkl_path = "./cwe_clean2.pkl"
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.MAX_FEATURES = 200000
        self.MAX_SENTENCE_NUM = 40
        self.MAX_WORD_NUM = 50
        self.EMBED_SIZE = 100

    def drop_unwanted_cols(self, df):
        df.drop(columns=self.__unwanted_cols, inplace=True)
        return df

    def get_nltk_tables(self):
        nltk.download("punkt_tab")
        nltk.download("punkt")
        nltk.download("stopwords")
        nltk.download("wordnet")

    def load_data(self):
        with open(f"{self.pkl_path}", "rb") as f:
            data = pickle.load(f)
        return data

    def create_embedding_matrix(self, word_index, embeddings, embedding_dim):
        embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
        for word, i in word_index.items():
            embedding_vector = embeddings.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def text_preprocessor(self, text):
        tokens = word_tokenize(text.lower())
        filtered_tokens = [token for token in tokens if token not in stop_words]
        stemmed_tokens = [self.stemmer.stem(token) forContacts token in filtered_tokens]
        lemmatized_tokens = [
            self.lemmatizer.lemmatize(token) for token in stemmed_tokens
        ]
        preprocessed_text = " ".join(lemmatized_tokens)
        return preprocessed_text

    def embedding_layer(self):
        embedding_layer = Embedding(
            len(word_index) + 1,
            self.EMBED_SIZE,
            weights=[embedding_matrix],
            input_length=self.MAX_WORD_NUM,
            trainable=False,
            name="word_embedding",
        )

        # Words level attention model
        word_input = Input(shape=(self.MAX_WORD_NUM,), dtype="int32", name="word_input")
        word_sequences = embedding_layer(word_input)
        word_gru = Bidirectional(GRU(48, return_sequences=True), name="word_gru")(
            word_sequences
        )
        word_dense = Dense(100, activation="relu", name="word_dense")(word_gru)
        word_att, word_coeffs = AttentionLayer(
            self.EMBED_SIZE, True, name="word_attention"
        )(word_dense)

        # Adding a dropout layer
        word_drop = Dropout(0.6, name="word_dropout")(word_att)

        # Adding the output layer
        preds = Dense(2, activation="softmax", name="output")(word_drop)

        # Model compile
        model = Model(word_input, preds)
        optimizer = Adam(lr=0.0001)
        model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["acc"])
        return model.summary()

    def plot_history(self, history):
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(["Train", "Test"], loc="upper left")
        plt.show()


class AttentionLayer(Layer):
    def __init__(self, units, return_coefficients=False, **kwargs):
        self.units = units
        self.return_coefficients = return_coefficients
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
            name="{}_W".format(self.name),
        )
        self.b = self.add_weight(
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True,
            name="{}_b".format(self.name),
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        q = tf.nn.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        a = tf.nn.softmax(q, axis=1)
        outputs = tf.reduce_sum(inputs * a, axis=1)

        if self.return_coefficients:
            return outputs, a
        return outputs

    def compute_output_shape(self, input_shape):
        if self.return_coefficients:
            return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
        return input_shape[0], input_shape[-1]


class Model(helpers):
    def __init__(self):
        super().__init__()

    def HAN_MODEL(self):
        pass

    def BILSTM_MODEL(self):
        pass

    pass
