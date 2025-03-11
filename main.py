#!/usr/bin/env python3

#!/usr/bin/env python3
"""
HAN (Hierarchical Attention Network) Classifier for CWE Data
Converted from Jupyter notebook: Copy_of_HAN20242lables.ipynb
"""

# Force CPU-only mode to avoid GPU errors
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Suppress TensorFlow logging except for errors
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    SpatialDropout1D,
    Bidirectional,
    GRU,
    Flatten,
)
from keras.layers import (
    TimeDistributed,
    concatenate,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
)
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import pickle
import copy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence


# Optional: Download necessary NLTK data: works better in some cases but not on this one
# Uncomment the following line if needed
# nltk.download('stopwords')


# Load data
def load_data(filepath):
    """
    Load the CWE data from a pickle or CSV file

    Args:
        filepath: Path to the data file

    Returns:
        DataFrame containing the CWE data
    """
    try:
        # Try if it's a CSV file
        if filepath.endswith(".csv"):
            cwe_data = pd.read_csv(filepath)
            print("Data loaded successfully from CSV!")
            print(cwe_data.head())
            return cwe_data

        # Try standard pickle loading
        cwe_data = pd.read_pickle(filepath)
        print("Data loaded successfully from pickle!")
        print(cwe_data.head())
        return cwe_data
    except Exception as e:
        print(f"Standard loading failed: {e}")
        try:
            # Fallback to a more robust loading method
            with open(filepath, "rb") as f:
                import pickle

                raw_data = pickle.load(f)

            # If it's already a DataFrame, use it
            if isinstance(raw_data, pd.DataFrame):
                cwe_data = raw_data
            else:
                # Otherwise try to construct a DataFrame
                cwe_data = pd.DataFrame(raw_data)

            print("Data loaded using fallback method!")
            print(cwe_data.head())
            return cwe_data
        except Exception as e2:
            print(f"Fallback loading failed: {e2}")
            print("Please convert your data to CSV format for more reliable loading.")
            return None


def preprocess_data(cwe_data):
    """
    Preprocess the data by dropping unwanted columns and filtering for specific labels

    Args:
        cwe_data: DataFrame containing the CWE data

    Returns:
        Preprocessed DataFrame
    """
    if cwe_data is None:
        return None

    # Deep copy for the next phase dropping unnecessary data columns
    cwe_deepcopy = copy.deepcopy(cwe_data)

    # Drop unwanted column
    cwe_deepcopy.drop(
        [
            "Authorization",
            "Accountability",
            "Authentication",
            "Non_Repudiation",
            "Availability",
            "Confidentiality",
            "Other",
        ],
        axis=1,
        inplace=True,
    )

    # Created a new df ignoring multi tags and only selecting with a single tag
    cwe_df = cwe_deepcopy

    # Convert list to string for easier comparison
    cwe_df["common_consequences_scope"] = cwe_df["common_consequences_scope"].apply(
        "".join
    )

    # Get index to drop the index which has more than 1 label
    dropIndex = cwe_df[
        (cwe_df["common_consequences_scope"] != "Access Control")
        & (cwe_df["common_consequences_scope"] != "Integrity")
    ].index

    # Drop the row by index if has more than 1 label
    cwe_df.drop(index=dropIndex, inplace=True)

    print(f"After preprocessing: {cwe_df.shape[0]} records")
    print(f"Unique labels: {cwe_df['common_consequences_scope'].unique()}")

    # Save in CSV format for more reliable future loading
    try:
        cwe_df.to_csv("cwe_clean2.csv", index=False)
        print("Data saved to CSV for future use")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    return cwe_df


def prepare_text_data(cwe_df, max_words=10000, max_sent_len=100, max_sent=15):
    """
    Prepare text data for HAN model by tokenizing and padding

    Args:
        cwe_df: Preprocessed DataFrame
        max_words: Maximum number of words to keep in the vocabulary
        max_sent_len: Maximum length of each sentence
        max_sent: Maximum number of sentences

    Returns:
        X_data: Processed text data
        y_data: Labels
        tokenizer: Fitted tokenizer
    """
    if cwe_df is None:
        return None, None, None

    # Use Clean_Description column as text data
    texts = cwe_df["Clean_Description"].values

    # Split texts into sentences and words
    # For HAN model, we need data in the format of [samples, sentences, words]
    documents = []
    for text in texts:
        # Simple sentence splitting (adjust as needed)
        sentences = text.split(". ")
        document = []
        for sentence in sentences[:max_sent]:
            words = sentence.split()
            document.append(words[:max_sent_len])
        # Pad document with empty sentences if needed
        while len(document) < max_sent:
            document.append([])
        documents.append(document)

    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    all_words = " ".join(texts)
    tokenizer.fit_on_texts([all_words])

    # Convert words to sequences
    data = np.zeros((len(documents), max_sent, max_sent_len), dtype="int32")
    for i, document in enumerate(documents):
        for j, sentence in enumerate(document):
            if sentence:
                seq = tokenizer.texts_to_sequences([" ".join(sentence)])[0]
                for k, word in enumerate(seq):
                    if k < max_sent_len:
                        data[i, j, k] = word

    # Prepare labels (convert to categorical)
    # Assuming 'Access_Control' and 'Integrity' are binary columns (0/1)
    labels = cwe_df[["Access_Control", "Integrity"]].values

    return data, labels, tokenizer


def build_han_model(
    max_words, max_sent_len, max_sent, embedding_dim=50, lstm_units=100
):
    """
    Build Hierarchical Attention Network model

    Args:
        max_words: Maximum vocabulary size
        max_sent_len: Maximum length of sentences
        max_sent: Maximum number of sentences
        embedding_dim: Dimension of embedding layer
        lstm_units: Number of LSTM units

    Returns:
        Compiled HAN model
    """
    # Word level
    word_input = Input(shape=(max_sent_len,), dtype="int32")
    word_embedding = Embedding(input_dim=max_words, output_dim=embedding_dim)(
        word_input
    )
    word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(word_embedding)
    word_dense = TimeDistributed(Dense(lstm_units * 2, activation="relu"))(word_lstm)
    word_att = TimeDistributed(Dense(1, activation="tanh"))(word_dense)
    word_att = Flatten()(word_att)
    word_att = tf.keras.layers.Activation("softmax")(word_att)
    word_att = tf.keras.layers.Reshape((max_sent_len, 1))(word_att)
    word_output = tf.keras.layers.Multiply()([word_lstm, word_att])
    word_output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(
        word_output
    )

    # Sentence level model
    sent_encoder = Model(word_input, word_output)

    # Document level
    document_input = Input(shape=(max_sent, max_sent_len), dtype="int32")
    document_encoder = TimeDistributed(sent_encoder)(document_input)
    document_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(
        document_encoder
    )
    document_dense = TimeDistributed(Dense(lstm_units * 2, activation="relu"))(
        document_lstm
    )
    document_att = TimeDistributed(Dense(1, activation="tanh"))(document_dense)
    document_att = Flatten()(document_att)
    document_att = tf.keras.layers.Activation("softmax")(document_att)
    document_att = tf.keras.layers.Reshape((max_sent, 1))(document_att)
    document_output = tf.keras.layers.Multiply()([document_lstm, document_att])
    document_output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(
        document_output
    )

    # Output layer
    output = Dense(2, activation="sigmoid")(document_output)

    # Final model
    model = Model(inputs=document_input, outputs=output)
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model


def train_model(model, X_data, y_data, epochs=10, batch_size=32, validation_split=0.2):
    """
    Train the HAN model

    Args:
        model: Compiled HAN model
        X_data: Processed text data
        y_data: Labels
        epochs: Number of epochs
        batch_size: Batch size
        validation_split: Validation split ratio

    Returns:
        Trained model and history
    """
    if X_data is None or y_data is None:
        return None, None

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    # Predict on test data
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Print confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred_binary.argmax(axis=1))
    print(cm)

    return model, history


def save_model(
    model,
    tokenizer,
    filepath="models/han_model.h5",
    tokenizer_path="models/tokenizer.pickle",
):
    """
    Save the model and tokenizer

    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        filepath: Path to save the model
        tokenizer_path: Path to save the tokenizer
    """
    if model is None:
        return

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the model
        model.save(filepath)
        print(f"Model saved to {filepath}")

        # Save the tokenizer
        with open(tokenizer_path, "wb") as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Tokenizer saved to {tokenizer_path}")
    except Exception as e:
        print(f"Error saving model and tokenizer: {e}")


def predict_text(model, tokenizer, text, max_sent_len=100, max_sent=15):
    """
    Make predictions on new text

    Args:
        model: Trained HAN model
        tokenizer: Fitted tokenizer
        text: Text to predict
        max_sent_len: Maximum length of sentences
        max_sent: Maximum number of sentences

    Returns:
        Prediction result
    """
    if model is None or tokenizer is None:
        return None

    # Preprocess the text
    sentences = text.split(". ")
    document = []
    for sentence in sentences[:max_sent]:
        words = sentence.split()
        document.append(words[:max_sent_len])
    # Pad document with empty sentences if needed
    while len(document) < max_sent:
        document.append([])

    # Convert words to sequences
    data = np.zeros((1, max_sent, max_sent_len), dtype="int32")
    for j, sentence in enumerate(document):
        if sentence:
            seq = tokenizer.texts_to_sequences([" ".join(sentence)])[0]
            for k, word in enumerate(seq):
                if k < max_sent_len:
                    data[0, j, k] = word

    # Make prediction
    prediction = model.predict(data)
    return prediction


def main():
    """
    Main function to run the HAN classifier
    """
    # Set parameters
    MAX_WORDS = 10000
    MAX_SENT_LEN = 100
    MAX_SENT = 15
    EMBEDDING_DIM = 100
    LSTM_UNITS = 100
    EPOCHS = 10
    BATCH_SIZE = 32

    print("Loading and preprocessing data...")
    # Use your dataset path here
    data_path = "./cwe_clean2.pkl"  # Replace with your actual path
    # Check if CSV version exists and use it preferentially
    if os.path.exists("./cwe_clean2.csv"):
        data_path = "./cwe_clean2.csv"
        print("Using CSV version of the data")

    # For demonstration purposes, we'll skip loading from a file
    cwe_data = load_data(data_path)
    cwe_df = preprocess_data(cwe_data)
    X_data, y_data, tokenizer = prepare_text_data(
        cwe_df, MAX_WORDS, MAX_SENT_LEN, MAX_SENT
    )

    print("Building and training the model...")
    model = build_han_model(
        MAX_WORDS, MAX_SENT_LEN, MAX_SENT, EMBEDDING_DIM, LSTM_UNITS
    )
    model, history = train_model(model, X_data, y_data, EPOCHS, BATCH_SIZE)

    print("Saving the model...")
    save_model(model, tokenizer)
    print("Done!")


if __name__ == "__main__":
    main()
