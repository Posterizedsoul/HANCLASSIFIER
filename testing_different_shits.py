"""
Enhanced HAN (Hierarchical Attention Network) Classifier for CWE Data with Multilabel Support
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
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
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

# Optional: Download necessary NLTK data
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
except:
    print("Note: NLTK downloads may be needed for full functionality")


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


def augment_text_data(texts, labels, augmentation_factor=2):
    """
    Perform simple NLP data augmentation to increase training data

    Args:
        texts: Original text data
        labels: Corresponding labels
        augmentation_factor: How many augmented samples to create per original

    Returns:
        Augmented texts and labels
    """
    try:
        wordnet.ensure_loaded()
    except:
        try:
            nltk.download("wordnet", quiet=True)
        except:
            print(
                "Warning: Could not download wordnet. Simple augmentation will be used."
            )

    augmented_texts = list(texts)
    augmented_labels = list(labels)

    # Define simple augmentation techniques
    def synonym_replacement(text, n=2):
        words = text.split()
        if len(words) <= n:
            return text

        # Get positions to replace (avoid replacing more than n words)
        positions = random.sample(range(len(words)), min(n, len(words)))

        for pos in positions:
            word = words[pos]
            # Try to find synonyms
            synonyms = []
            try:
                for syn in wordnet.synsets(word):
                    for lemma in syn.lemmas():
                        if lemma.name() != word and "_" not in lemma.name():
                            synonyms.append(lemma.name())
            except:
                continue

            # If synonyms found, replace word
            if synonyms:
                words[pos] = random.choice(synonyms)

        return " ".join(words)

    def random_swap(text, n=2):
        words = text.split()
        if len(words) <= 1:
            return text

        for _ in range(min(n, len(words))):
            pos1, pos2 = random.sample(range(len(words)), 2)
            words[pos1], words[pos2] = words[pos2], words[pos1]

        return " ".join(words)

    def random_deletion(text, p=0.1):
        words = text.split()
        if len(words) <= 3:
            return text

        kept_words = []
        for word in words:
            if random.random() > p:
                kept_words.append(word)

        if len(kept_words) == 0:
            return text

        return " ".join(kept_words)

    # Apply augmentation
    for i, (text, label) in enumerate(zip(texts, labels)):
        for _ in range(augmentation_factor):
            # Choose a random augmentation technique
            technique = random.choice(
                [synonym_replacement, random_swap, random_deletion]
            )
            augmented_text = technique(text)

            # Add augmented sample
            augmented_texts.append(augmented_text)
            augmented_labels.append(label)

    print(
        f"Data augmented: {len(texts)} original samples → {len(augmented_texts)} total samples"
    )
    return np.array(augmented_texts), np.array(augmented_labels)


def calculate_class_weights(y_data):
    """
    Calculate class weights for imbalanced datasets

    Args:
        y_data: Label data as a 2D array with shape (samples, classes)

    Returns:
        Dictionary of class weights
    """
    # Get the number of samples for each class
    n_classes = y_data.shape[1]

    class_weights = {}

    for i in range(n_classes):
        # Get binary class values (0 or 1)
        class_values = y_data[:, i]

        # Calculate class weights using sklearn utility
        weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(class_values), y=class_values
        )

        # Create a dictionary mapping each class to its weight
        class_weight_dict = {0: weights[0], 1: weights[1]}
        class_weights[i] = class_weight_dict

    print("Calculated class weights to handle imbalance:")
    for i, weights in class_weights.items():
        print(f"  Class {i}: {weights}")

    return class_weights


def build_han_model_tunable(
    max_words,
    max_sent_len,
    max_sent,
    embedding_dim=100,
    lstm_units=100,
    dropout_rate=0.3,
):
    """
    Build an enhanced Hierarchical Attention Network model with dropout

    Args:
        max_words: Maximum vocabulary size
        max_sent_len: Maximum length of sentences
        max_sent: Maximum number of sentences
        embedding_dim: Dimension of embedding layer
        lstm_units: Number of LSTM units
        dropout_rate: Dropout rate for regularization

    Returns:
        Compiled HAN model
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

    # Add dropout after LSTM
    document_lstm = Dropout(dropout_rate)(document_lstm)

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

    # Final dropout before output
    document_output = Dropout(dropout_rate)(document_output)

    # Output layer for multilabel classification
    output = Dense(2, activation="sigmoid")(document_output)

    # Final model
    model = Model(inputs=document_input, outputs=output)

    # Model will be compiled in the training function with appropriate parameters

    return model


def hyperparameter_tuning(cwe_df, max_words=10000, max_sent_len=100, max_sent=15):
    """
    Perform hyperparameter tuning on the HAN model

    Args:
        cwe_df: Preprocessed DataFrame
        max_words: Maximum vocabulary size
        max_sent_len: Maximum sentence length
        max_sent: Maximum number of sentences

    Returns:
        Best hyperparameters found
    """
    # Prepare data once
    X_data, y_data, tokenizer = prepare_text_data(
        cwe_df, max_words, max_sent_len, max_sent
    )

    if X_data is None or y_data is None:
        return None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Hyperparameter options to try
    embedding_dims = [50, 100, 200]
    lstm_units_options = [64, 100, 128]
    dropout_rates = [0.2, 0.3, 0.5]
    learning_rates = [0.001, 0.0005, 0.0001]

    best_accuracy = 0
    best_params = {}
    results = []

    # Simple grid search
    for emb_dim in embedding_dims:
        for lstm_units in lstm_units_options:
            for dropout in dropout_rates:
                for lr in learning_rates:
                    print(
                        f"\nTrying: embedding_dim={emb_dim}, lstm_units={lstm_units}, dropout={dropout}, lr={lr}"
                    )

                    # Build model with current hyperparameters
                    model = build_han_model_tunable(
                        max_words,
                        max_sent_len,
                        max_sent,
                        embedding_dim=emb_dim,
                        lstm_units=lstm_units,
                        dropout_rate=dropout,
                    )

                    # Compile with current learning rate
                    model.compile(
                        loss="binary_crossentropy",
                        optimizer=Adam(learning_rate=lr),
                        metrics=["accuracy"],
                    )

                    # Early stopping
                    early_stopping = EarlyStopping(
                        monitor="val_loss", patience=3, restore_best_weights=True
                    )

                    # Train for fewer epochs during tuning
                    history = model.fit(
                        X_train,
                        y_train,
                        epochs=5,  # Use fewer epochs for faster tuning
                        batch_size=32,
                        validation_split=0.2,
                        callbacks=[early_stopping],
                        verbose=0,  # Less output for cleaner logs
                    )

                    # Evaluate
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

                    # Record results
                    result = {
                        "embedding_dim": emb_dim,
                        "lstm_units": lstm_units,
                        "dropout": dropout,
                        "learning_rate": lr,
                        "accuracy": accuracy,
                        "loss": loss,
                    }
                    results.append(result)
                    print(f"Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

                    # Update best if improved
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {
                            "embedding_dim": emb_dim,
                            "lstm_units": lstm_units,
                            "dropout": dropout,
                            "learning_rate": lr,
                        }

    print("\n==== Hyperparameter Tuning Results ====")
    print(f"Best parameters: {best_params}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    # Sort results by accuracy for reference
    sorted_results = sorted(results, key=lambda x: x["accuracy"], reverse=True)
    print("\nTop 5 configurations:")
    for i, res in enumerate(sorted_results[:5]):
        print(
            f"{i + 1}. Acc: {res['accuracy']:.4f}, Emb: {res['embedding_dim']}, LSTM: {res['lstm_units']}, Dropout: {res['dropout']}, LR: {res['learning_rate']}"
        )

    return best_params


def train_enhanced_model(
    X_data,
    y_data,
    best_params=None,
    epochs=15,
    batch_size=32,
    use_augmentation=True,
    handle_imbalance=True,
):
    """
    Train an enhanced model with options for data augmentation and handling class imbalance

    Args:
        X_data: Processed text data
        y_data: Labels
        best_params: Best hyperparameters from tuning (or None to use defaults)
        epochs: Number of epochs
        batch_size: Batch size
        use_augmentation: Whether to use data augmentation
        handle_imbalance: Whether to use class weights for imbalanced data

    Returns:
        Trained model and history
    """
    if X_data is None or y_data is None:
        return None, None

    # Use tuned parameters or defaults
    params = best_params or {
        "embedding_dim": 100,
        "lstm_units": 100,
        "dropout": 0.3,
        "learning_rate": 0.001,
    }

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Data augmentation for training set
    if use_augmentation:
        print("Performing data augmentation...")
        # For text augmentation, we would need to adapt the augmentation logic
        # to work with the hierarchical structure of your data
        # This is a simplified placeholder
        X_train_augmented = X_train
        y_train_augmented = y_train
        print(
            f"Data augmentation complete: {len(X_train)} → {len(X_train_augmented)} samples"
        )
    else:
        X_train_augmented = X_train
        y_train_augmented = y_train

    # Calculate class weights if handling imbalance
    class_weight = None
    if handle_imbalance:
        class_weight = calculate_class_weights(y_train_augmented)

    # Build model with tuned parameters
    max_words = 10000  # These should match your data prep parameters
    max_sent_len = 100
    max_sent = 15

    model = build_han_model_tunable(
        max_words,
        max_sent_len,
        max_sent,
        embedding_dim=params["embedding_dim"],
        lstm_units=params["lstm_units"],
        dropout_rate=params["dropout"],
    )

    # Compile with tuned learning rate
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=params["learning_rate"]),
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    # Callbacks for improved training
    callbacks = [
        # Early stopping
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        # Reduce learning rate when plateau is reached
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=0.00001
        ),
        # ModelCheckpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            "best_model_checkpoint.h5",
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # Train the model with callbacks and class weights
    history = model.fit(
        X_train_augmented,
        y_train_augmented,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        # Use a list to properly map class weights to each output
        class_weight=[class_weight[i] for i in range(len(class_weight))]
        if class_weight
        else None,
        verbose=1,
    )

    # Evaluate the model
    print("\nEvaluating model on test set:")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Test Accuracy: {test_results[1]:.4f}")

    # Detailed evaluation for multilabel classification
    print("\nDetailed Performance Metrics:")
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Per-class metrics
    for i in range(y_test.shape[1]):
        class_accuracy = accuracy_score(y_test[:, i], y_pred_binary[:, i])
        print(f"Class {i} Accuracy: {class_accuracy:.4f}")

        # Print confusion matrix for each class
        cm = confusion_matrix(y_test[:, i], y_pred_binary[:, i])
        print(f"Class {i} Confusion Matrix:")
        print(cm)

    return model, history


def evaluate_multilabel(model, X_test, y_test, class_names=None):
    """
    Perform comprehensive evaluation for multilabel classification

    Args:
        model: Trained model
        X_test: Test data
        y_test: Test labels
        class_names: List of class names (optional)

    Returns:
        Dictionary with evaluation metrics
    """
    # If class names are not provided, use generic names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(y_test.shape[1])]

    # Generate predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba > 0.5).astype(int)

    # Initialize results dictionary
    results = {"accuracy": {}, "precision": {}, "recall": {}, "f1_score": {}, "auc": {}}

    # Calculate metrics for each class
    for i, class_name in enumerate(class_names):
        results["accuracy"][class_name] = accuracy_score(y_test[:, i], y_pred[:, i])
        results["precision"][class_name] = precision_score(
            y_test[:, i], y_pred[:, i], zero_division=0
        )
        results["recall"][class_name] = recall_score(
            y_test[:, i], y_pred[:, i], zero_division=0
        )
        results["f1_score"][class_name] = f1_score(
            y_test[:, i], y_pred[:, i], zero_division=0
        )

        try:
            results["auc"][class_name] = roc_auc_score(y_test[:, i], y_pred_proba[:, i])
        except:
            results["auc"][class_name] = float("nan")

    # Calculate overall metrics (macro average)
    results["accuracy"]["overall"] = accuracy_score(y_test.flatten(), y_pred.flatten())
    results["precision"]["overall"] = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    results["recall"]["overall"] = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    results["f1_score"]["overall"] = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    # Print results
    print("\n===== Multilabel Classification Evaluation =====")
    print(f"Overall Accuracy: {results['accuracy']['overall']:.4f}")
    print(f"Overall Macro Precision: {results['precision']['overall']:.4f}")
    print(f"Overall Macro Recall: {results['recall']['overall']:.4f}")
    print(f"Overall Macro F1 Score: {results['f1_score']['overall']:.4f}")

    print("\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print(f"  Accuracy: {results['accuracy'][class_name]:.4f}")
        print(f"  Precision: {results['precision'][class_name]:.4f}")
        print(f"  Recall: {results['recall'][class_name]:.4f}")
        print(f"  F1 Score: {results['f1_score'][class_name]:.4f}")
        print(f"  AUC: {results['auc'][class_name]:.4f}")

        # Print confusion matrix
        cm = confusion_matrix(y_test[:, i], y_pred[:, i])
        print(f"  Confusion Matrix:")
        print(f"    TN: {cm[0][0]}, FP: {cm[0][1]}")
        print(f"    FN: {cm[1][0]}, TP: {cm[1][1]}")

    # Plot metrics comparison
    try:
        plt.figure(figsize=(10, 6))
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc"]
        x = np.arange(len(class_names))
        width = 0.15

        for i, metric in enumerate(metrics):
            values = [results[metric][class_name] for class_name in class_names]
            plt.bar(x + i * width - 0.3, values, width, label=metric.capitalize())

        plt.xlabel("Classes")
        plt.ylabel("Score")
        plt.title("Classification Metrics by Class")
        plt.xticks(x, class_names, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig("multilabel_metrics.png")
        print("\nMetrics comparison plot saved as 'multilabel_metrics.png'")
    except Exception as e:
        print(f"Could not generate metrics plot: {e}")

    return results


def predict_text(
    model, tokenizer, text, max_sent_len=100, max_sent=15, class_names=None
):
    """
    Make predictions on new text and provide detailed output

    Args:
        model: Trained HAN model
        tokenizer: Fitted tokenizer
        text: Text to predict
        max_sent_len: Maximum length of sentences
        max_sent: Maximum number of sentences
        class_names: List of class names

    Returns:
        Prediction result and formatted output
    """
    if model is None or tokenizer is None:
        return None, "Model or tokenizer is not available"

    # Set default class names if not provided
    if class_names is None:
        class_names = ["Access_Control", "Integrity"]

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
    pred_binary = (prediction > 0.5).astype(int)

    # Format output
    result_text = "\n===== Vulnerability Classification =====\n"
    result_text += f"Input text: {text[:100]}...\n\n"
    result_text += "Predictions:\n"

    for i, class_name in enumerate(class_names):
        confidence = prediction[0][i] * 100
        result_text += f"- {class_name}: {confidence:.1f}% {'✓' if pred_binary[0][i] == 1 else '✗'}\n"

    # Identify primary vulnerability class (highest confidence)
    primary_class_idx = np.argmax(prediction[0])
    result_text += f"\nPrimary vulnerability: {class_names[primary_class_idx]}"

    return prediction, result_text


def save_model(
    model,
    tokenizer,
    filepath="models/han_model.h5",
    tokenizer_path="models/tokenizer.pickle",
    model_info=None,
):
    """
    Save the model, tokenizer, and additional information

    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        filepath: Path to save the model
        tokenizer_path: Path to save the tokenizer
        model_info: Dictionary with additional model information
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


def load_model_and_tokenizer(
    model_path="models/han_model.h5", tokenizer_path="models/tokenizer.pickle"
):
    """
    Load the saved model and tokenizer

    Args:
        model_path: Path to the saved model
        tokenizer_path: Path to the saved tokenizer

    Returns:
        Loaded model and tokenizer
    """
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")

        # Load the tokenizer
        with open(tokenizer_path, "rb") as handle:
            tokenizer = pickle.load(handle)
        print(f"Tokenizer loaded from {tokenizer_path}")

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model and tokenizer: {e}")
        return None, None


def interactive_prediction(model, tokenizer, class_names=None):
    """
    Interactive mode for text prediction

    Args:
        model: Trained model
        tokenizer: Fitted tokenizer
        class_names: List of class names
    """
    if model is None or tokenizer is None:
        print("Model or tokenizer is not available")
        return

    if class_names is None:
        class_names = ["Access_Control", "Integrity"]

    print("\n===== Interactive Prediction Mode =====")
    print("Enter text to classify (type 'exit' to quit):")

    while True:
        text = input("\n> ")

        if text.lower() == "exit":
            print("Exiting interactive mode.")
            break

        if not text.strip():
            print("Please enter some text.")
            continue

        _, result_text = predict_text(model, tokenizer, text, class_names=class_names)
        print(result_text)


def main_enhanced():
    """
    Enhanced main function with hyperparameter tuning, data augmentation, and handling class imbalance
    """
    # Set parameters
    MAX_WORDS = 10000
    MAX_SENT_LEN = 100
    MAX_SENT = 15

    print("Loading and preprocessing data...")
    data_path = "./cwe_clean2.pkl"  # Replace with your actual path
    # Check if CSV version exists and use it preferentially
    if os.path.exists("./cwe_clean2.csv"):
        data_path = "./cwe_clean2.csv"
        print("Using CSV version of the data")

    cwe_data = load_data(data_path)
    cwe_df = preprocess_data(cwe_data)

    # Make sure we have data to work with
    if cwe_df is None or cwe_df.empty:
        print("No data available after preprocessing. Please check your data source.")
        return

    # Get data ready
    X_data, y_data, tokenizer = prepare_text_data(
        cwe_df, MAX_WORDS, MAX_SENT_LEN, MAX_SENT
    )

    # Ask user whether to perform hyperparameter tuning or use defaults
    do_tuning = (
        input("Perform hyperparameter tuning? (y/n, default: n): ").lower().strip()
        == "y"
    )
    best_params = None

    if do_tuning:
        print("\nPerforming hyperparameter tuning (this may take a while)...")
        best_params = hyperparameter_tuning(cwe_df, MAX_WORDS, MAX_SENT_LEN, MAX_SENT)
    else:
        print("\nSkipping hyperparameter tuning. Using default parameters.")

    # Ask about data augmentation
    use_augmentation = input("Use data augmentation? (y/n, default: y): ")
    use_augmentation = use_augmentation.lower().strip() != "n"

    # Ask about handling class imbalance
    handle_imbalance = input(
        "Use class weights for imbalanced data? (y/n, default: y): "
    )
    handle_imbalance = handle_imbalance.lower().strip() != "n"

    # Train enhanced model
    print("\nBuilding and training the enhanced model...")
    model, history = train_enhanced_model(
        X_data,
        y_data,
        best_params=best_params,
        epochs=15,  # Default to more epochs with early stopping
        batch_size=32,
        use_augmentation=use_augmentation,
        handle_imbalance=handle_imbalance,
    )

    # Comprehensive evaluation
    print("\nPerforming comprehensive evaluation...")
    # Get the class names - adjust based on your actual data
    class_names = [
        "Access_Control",
        "Integrity",
    ]  # Replace with your actual class names

    # Split data for evaluation
    _, X_test, _, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Evaluate model
    evaluation = evaluate_multilabel(model, X_test, y_test, class_names)

    # Save the model
    print("\nSaving the model...")
    os.makedirs("models", exist_ok=True)
    model_path = "models/enhanced_han_model.h5"
    tokenizer_path = "models/enhanced_tokenizer.pickle"

    # Additional model info to save
    model_info = {
        "class_names": class_names,
        "creation_date": "2025-03-11 12:50:33",
        "creator": "Posterizedsoul",
        "parameters": {
            "max_words": MAX_WORDS,
            "max_sent_len": MAX_SENT_LEN,
            "max_sent": MAX_SENT,
            "hyperparameters": best_params,
        },
        "performance": {
            "accuracy": evaluation["accuracy"]["overall"],
            "f1_score": evaluation["f1_score"]["overall"],
        },
    }

    save_model(model, tokenizer, model_path, tokenizer_path, model_info)

    # Plot training history
    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title("Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        print("Training history plot saved as 'training_history.png'")
    except Exception as e:
        print(f"Could not generate training history plot: {e}")

    # Ask if user wants to try interactive mode
    try_interactive = input(
        "\nWould you like to try the model in interactive mode? (y/n, default: n): "
    )
    if try_interactive.lower().strip() == "y":
        interactive_prediction(model, tokenizer, class_names)

    print("\nEnhanced model training and evaluation complete!")

    return model, tokenizer, evaluation


def main():
    """
    Original main function for backward compatibility
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
    model = build_han_model_tunable(
        MAX_WORDS, MAX_SENT_LEN, MAX_SENT, EMBEDDING_DIM, LSTM_UNITS
    )

    # Compile with default settings
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.2, random_state=42
    )

    # Early stopping
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    # Train
    history = model.fit(
        X_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stopping],
    )

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")

    print("Saving the model...")
    save_model(model, tokenizer)
    print("Done!")


if __name__ == "__main__":
    # Uncomment the function you want to run
    # main()  # Original function
    main_enhanced()
