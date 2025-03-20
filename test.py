import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
    TimeDistributed,
    concatenate,
    GlobalMaxPooling1D,
    GlobalAveragePooling1D,
    LayerNormalization,
    MultiHeadAttention,
)
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import copy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
import random
import gc
import psutil
from multiprocessing import cpu_count

# Download NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("punkt_tab", quiet=True)


# Load pre-trained GloVe embeddings
def load_glove_embeddings(embedding_dim=100):
    embeddings_index = {}
    with open(f"glove.6B.{embedding_dim}d.txt", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs
    return embeddings_index


# Create embedding matrix
def create_embedding_matrix(tokenizer, embeddings_index, max_words, embedding_dim):
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < max_words:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


# Focal loss for imbalanced data
def focal_loss(gamma=2.0, alpha=0.25):
    def loss(y_true, y_pred):
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return -tf.reduce_sum(alpha * tf.pow(1.0 - pt, gamma) * tf.math.log(pt + 1e-7))

    return loss


def preprocess_data(cwe_data):
    """
    Preprocess the data by dropping unwanted columns and filtering for specific labels.

    Args:
        cwe_data: DataFrame containing the CWE data.

    Returns:
        Preprocessed DataFrame.
    """
    if cwe_data is None:
        return None

    # Deep copy for the next phase
    cwe_deepcopy = copy.deepcopy(cwe_data)

    # Only drop columns that exist in the dataframe
    columns_to_drop = [
        "Authorization",
        "Accountability",
        "Authentication",
        "Non_Repudiation",
        "Availability",
        "Confidentiality",
        "Other",
    ]

    # Filter the list to only include columns that actually exist
    existing_columns = [col for col in columns_to_drop if col in cwe_deepcopy.columns]

    # Only attempt to drop if there are columns to drop
    if existing_columns:
        cwe_deepcopy.drop(existing_columns, axis=1, inplace=True)

    # Created a new df ignoring multi tags and only selecting with a single tag
    cwe_df = cwe_deepcopy

    # Make sure the column exists before processing
    if "common_consequences_scope" in cwe_df.columns:
        # Convert list to string for easier comparison
        cwe_df["common_consequences_scope"] = cwe_df["common_consequences_scope"].apply(
            lambda x: "".join(x) if isinstance(x, list) else x
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
    else:
        print("Warning: 'common_consequences_scope' column not found in dataframe")

    # Save in CSV format for more reliable future loading
    try:
        cwe_df.to_csv("cwe_clean2.csv", index=False)
        print("Data saved to CSV for future use")
    except Exception as e:
        print(f"Error saving CSV: {e}")

    return cwe_df


def prepare_text_data(cwe_df, max_words=10000, max_sent_len=100, max_sent=15):
    """
    Prepare text data for HAN model by tokenizing and padding.

    Args:
        cwe_df: Preprocessed DataFrame.
        max_words: Maximum number of words to keep in the vocabulary.
        max_sent_len: Maximum length of each sentence.
        max_sent: Maximum number of sentences.

    Returns:
        X_data: Processed text data.
        y_data: Labels.
        tokenizer: Fitted tokenizer.
    """
    if cwe_df is None:
        return None, None, None

    # Use Clean_Description column as text data
    texts = cwe_df["Clean_Description"].values

    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    # Split texts into sentences and words
    # For HAN model, we need data in the format of [samples, sentences, words]
    documents = []
    for text in texts:
        # Preprocess text (lemmatization and stopword removal)
        processed_sentences = preprocess_text(text, lemmatizer, stop_words)
        document = []
        for sentence in processed_sentences[:max_sent]:
            words = sentence[:max_sent_len]
            document.append(words)
        # Pad document with empty sentences if needed
        while len(document) < max_sent:
            document.append([])
        documents.append(document)

    # Create and fit tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    all_words = " ".join(
        [" ".join(sentence) for document in documents for sentence in document]
    )
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


def save_model(
    model,
    tokenizer,
    filepath="models/han_model.h5",
    tokenizer_path="models/tokenizer.pickle",
    model_info=None,
):
    """
    Save the model, tokenizer, and additional information.

    Args:
        model: Trained model.
        tokenizer: Fitted tokenizer.
        filepath: Path to save the model.
        tokenizer_path: Path to save the tokenizer.
        model_info: Dictionary with additional model information.
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

        # Save additional model information if provided
        if model_info:
            info_path = os.path.join(os.path.dirname(filepath), "model_info.pkl")
            with open(info_path, "wb") as handle:
                pickle.dump(model_info, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Model information saved to {info_path}")

    except Exception as e:
        print(f"Error saving model and tokenizer: {e}")


def augment_hierarchical_data(documents, labels, augmentation_factor=2):
    """
    Perform hierarchical data augmentation on tokenized sequences.

    Args:
        documents: Tokenized documents (3D array of shape [samples, sentences, words]).
        labels: Corresponding labels.
        augmentation_factor: How many augmented samples to create per original.

    Returns:
        Augmented documents and labels.
    """
    augmented_docs = []
    augmented_labels = []

    for doc, label in zip(documents, labels):
        for _ in range(augmentation_factor):
            # Apply augmentation to each sentence in the document
            new_doc = []
            for sentence in doc:
                # Skip empty sentences
                if np.sum(sentence) == 0:
                    new_doc.append(sentence)
                    continue

                # Randomly swap words in the sentence
                if random.random() < 0.5:
                    non_zero_indices = np.where(sentence != 0)[0]
                    if len(non_zero_indices) > 1:
                        idx1, idx2 = random.sample(list(non_zero_indices), 2)
                        sentence[idx1], sentence[idx2] = sentence[idx2], sentence[idx1]

                # Randomly delete words in the sentence
                if random.random() < 0.3:
                    non_zero_indices = np.where(sentence != 0)[0]
                    if len(non_zero_indices) > 1:
                        delete_indices = random.sample(
                            list(non_zero_indices), min(1, len(non_zero_indices) // 2)
                        )
                        sentence[delete_indices] = 0

                new_doc.append(sentence)

            augmented_docs.append(new_doc)
            augmented_labels.append(label)

    print(
        f"Data augmented: {len(documents)} original samples → {len(augmented_docs)} total samples"
    )
    return np.array(augmented_docs), np.array(augmented_labels)


# Enhanced preprocessing with lemmatization
def preprocess_text(text, lemmatizer, stop_words):
    """
    Preprocess text by tokenizing, lemmatizing, and removing stopwords.

    Args:
        text: Input text.
        lemmatizer: WordNetLemmatizer instance.
        stop_words: Set of stopwords.

    Returns:
        List of processed sentences.
    """
    sentences = sent_tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        words = word_tokenize(sentence)
        words = [
            lemmatizer.lemmatize(word)
            for word in words
            if word.isalnum() and word not in stop_words
        ]
        processed_sentences.append(words)
    return processed_sentences


def build_han_model_tunable(
    max_words,
    max_sent_len,
    max_sent,
    embedding_dim=100,
    lstm_units=100,
    dropout_rate=0.3,
):
    """
    Build an enhanced Hierarchical Attention Network model with dropout.

    Args:
        max_words: Maximum vocabulary size.
        max_sent_len: Maximum length of sentences.
        max_sent: Maximum number of sentences.
        embedding_dim: Dimension of embedding layer.
        lstm_units: Number of LSTM units.
        dropout_rate: Dropout rate for regularization.

    Returns:
        Compiled HAN model.
    """
    # Word level
    word_input = Input(shape=(max_sent_len,), dtype="int32")
    word_embedding = Embedding(input_dim=max_words, output_dim=embedding_dim)(
        word_input
    )

    # Add dropout after embedding
    word_embedding = SpatialDropout1D(dropout_rate)(word_embedding)

    word_lstm = Bidirectional(LSTM(lstm_units, return_sequences=True))(word_embedding)
    word_dense = TimeDistributed(Dense(lstm_units * 2, activation="relu"))(word_lstm)

    # Add dropout after dense layer
    word_dense = TimeDistributed(Dropout(dropout_rate))(word_dense)

    # Attention mechanism
    word_att = TimeDistributed(Dense(1, activation="tanh"))(word_dense)
    word_att = Flatten()(word_att)
    word_att = tf.keras.layers.Activation("softmax")(word_att)

    # Reshape to match the original sequence length
    word_att = tf.keras.layers.Reshape((max_sent_len, 1))(word_att)

    # Apply attention weights
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

    # Add dropout after LSTM
    document_lstm = Dropout(dropout_rate)(document_lstm)

    document_dense = TimeDistributed(Dense(lstm_units * 2, activation="relu"))(
        document_lstm
    )

    # Document-level attention
    document_att = TimeDistributed(Dense(1, activation="tanh"))(document_dense)
    document_att = Flatten()(document_att)
    document_att = tf.keras.layers.Activation("softmax")(document_att)

    # Reshape to match the original sequence length
    document_att = tf.keras.layers.Reshape((max_sent, 1))(document_att)

    # Apply attention weights
    document_output = tf.keras.layers.Multiply()([document_lstm, document_att])
    document_output = tf.keras.layers.Lambda(lambda x: tf.keras.backend.sum(x, axis=1))(
        document_output
    )

    # Final dropout before output
    document_output = Dropout(dropout_rate)(document_output)

    # Output layer for multilabel classification
    output = Dense(2, activation="sigmoid")(document_output)

    # Final model
    model = Model(inputs=document_input, outputs=output)

    return model


# Main training function
def train_enhanced_model(
    X_data,
    y_data,
    best_params=None,
    epochs=15,
    batch_size=32,
    use_augmentation=True,
    handle_imbalance=True,
):
    if X_data is None or y_data is None:
        return None, None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Data augmentation
    if use_augmentation:
        X_train, y_train = augment_hierarchical_data(X_train, y_train)

    # Build model
    model = build_han_model_tunable(
        max_words=10000,
        max_sent_len=100,
        max_sent=15,
        embedding_dim=best_params.get("embedding_dim", 100),
        lstm_units=best_params.get("lstm_units", 100),
        dropout_rate=best_params.get("dropout", 0.3),
    )

    # Compile with focal loss
    model.compile(
        loss=focal_loss(),
        optimizer=Adam(learning_rate=best_params.get("learning_rate", 0.001)),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
        ],
    )

    # Evaluate
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

    return model, history


# Main function
def main():
    # Load and preprocess data
    data_path = "./cwe_clean2.csv"  # Replace with your dataset path
    cwe_data = pd.read_csv(data_path)
    cwe_df = preprocess_data(cwe_data)

    # Prepare text data
    X_data, y_data, tokenizer = prepare_text_data(
        cwe_df, max_words=10000, max_sent_len=100, max_sent=15
    )

    # Train the enhanced model
    model, history = train_enhanced_model(
        X_data,
        y_data,
        best_params={
            "embedding_dim": 100,
            "lstm_units": 128,
            "dropout": 0.3,
            "learning_rate": 0.001,
        },
        epochs=15,
        batch_size=32,
        use_augmentation=True,
        handle_imbalance=True,
    )

    # Save the model
    save_model(
        model,
        tokenizer,
        model_path="models/enhanced_han_model.h5",
        tokenizer_path="models/enhanced_tokenizer.pickle",
    )


if __name__ == "__main__":
    main()
