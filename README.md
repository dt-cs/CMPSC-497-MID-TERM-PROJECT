# Command Classification with CNN and GloVe Embeddings

This project implements a Convolutional Neural Network (CNN) model for classifying natural language queries into command categories. It uses pre-trained GloVe word embeddings for text representation.

## Project Overview

The system takes natural language queries as input and classifies them into appropriate command categories. It's primarily designed for command classification in systems like Rhino 3D, but the approach can be adapted for other command classification tasks.

## Key Features

- **Text Classification CNN**: Implements a CNN architecture optimized for text classification
- **Pre-trained Word Embeddings**: Uses GloVe embeddings for rich word representation (100-d)
- **Out-of-Domain Detection**: Identifies queries that don't belong to any known category
- **Performance Visualization**: Includes tools to visualize model performance and confusion matrices

## Project Structure

- **command_classification_annotated.ipynb**: Main notebook with detailed annotations of all steps

## Model Architecture

The model follows this pipeline:
1. Text preprocessing (lowercase, tokenization, etc.)
2. Word embedding lookup using GloVe
3. Multiple parallel convolutions with different filter sizes of [3,4]
4. Max-over-time pooling
5. Concatenation of pooled features
6. Dropout for regularization
7. Fully connected layer for classification

## Performance

The model achieves high accuracy on command classification tasks. Evaluation metrics include:
- Accuracy: 90%+ on test set
- Detailed per-category precision and recall
- Confusion matrix for error analysis

## Requirements

- Python 3.6+
- PyTorch
- NLTK
- scikit-learn
- gensim
- matplotlib
- seaborn
- tqdm

## Usage

1. Prepare your data in JSONL format with 'query' and 'category' fields
2. Run the notebook or use the standalone Python files
3. Evaluate model performance using the provided visualization tools
