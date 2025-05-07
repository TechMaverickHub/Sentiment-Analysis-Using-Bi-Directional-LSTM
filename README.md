# Sentiment Analysis Using Bi-Directional LSTM

This project performs binary sentiment analysis on the IMDb movie reviews dataset using a deep learning model based on a Bi-Directional Long Short-Term Memory (BiLSTM) network.

## 📝 Overview

The goal of this project is to classify movie reviews as **positive** or **negative** using deep learning. It demonstrates text preprocessing, neural network modeling, training, and evaluation using the TensorFlow/Keras ecosystem.

## 📊 Dataset

- **Source**: Built-in `imdb` dataset from Keras.
- **Samples**: 25,000 training and 25,000 testing examples.
- **Classes**: Binary classification — Positive (1) or Negative (0).

## 🧹 Preprocessing

- Vocabulary limited to the top 10,000 words.
- Reviews are padded/truncated to a maximum length of 200 tokens.
- Text sequences converted to human-readable format for inspection.
- Data split into training, validation, and testing sets.

## 🧠 Model Architecture

- `Embedding` layer for word vector representation.
- `Bidirectional(LSTM)` to capture context in both directions.
- `Dropout` for regularization.
- `Dense` output layer with sigmoid activation for binary classification.

```python
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=200),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
