  #!/usr/bin/env python3
"""
RoBERTa-based Financial Sentiment Analysis

This module provides functionality for training and evaluating a sentiment analysis
model using RoBERTa transformer on financial text data.

Author: Paarth Rattan
"""

import os
import re
import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Constants
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MAX_LEN = 140
MIN_WORDS = 4

# Pretrained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
roberta_model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def strip_all_entities(text):
    """Remove URLs, mentions, non-ASCII characters, and punctuation from text."""
    text = text.replace('\r', '').replace('\n', ' ').lower()
    text = re.sub(r"(?:@|https?://)\S+", "", text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    banned_chars = string.punctuation + 'Ã±ã¼â»§'
    return text.translate(str.maketrans('', '', banned_chars))

def clean_hashtags(text):
    """Remove '#' from hashtags."""
    return ' '.join(word[1:] if word.startswith('#') else word for word in text.split())

def filter_special_chars(text):
    """Remove words containing $ or &."""
    return ' '.join('' if ('$' in w or '&' in w) else w for w in text.split())

def remove_extra_spaces(text):
    """Collapse multiple spaces into a single space."""
    return re.sub(r"\s{2,}", " ", text).strip()

def filter_by_word_count(text, min_words=MIN_WORDS):
    """Keep only texts with minimum number of words."""
    return len(text.split()) >= min_words

def preprocess_text(df, text_col='Sentence'):
    """Apply text cleaning functions to DataFrame."""
    df = df.copy()
    df['text_clean'] = df[text_col].astype(str)
    df['text_clean'] = df['text_clean'].apply(strip_all_entities)
    df['text_clean'] = df['text_clean'].apply(clean_hashtags)
    df['text_clean'] = df['text_clean'].apply(filter_special_chars)
    df['text_clean'] = df['text_clean'].apply(remove_extra_spaces)
    
    # Filter by word count
    word_count_mask = df['text_clean'].apply(filter_by_word_count)
    df = df[word_count_mask].reset_index(drop=True)
    
    print(f"Kept {len(df)} samples with {MIN_WORDS}+ words")
    return df

def tokenize_data(texts, max_len=MAX_LEN):
    """Tokenize texts for RoBERTa model."""
    input_ids, attention_masks = [], []
    for txt in texts:
        enc = tokenizer.encode_plus(
            txt,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(enc['input_ids'])
        attention_masks.append(enc['attention_mask'])
    return np.array(input_ids), np.array(attention_masks)

def build_model(max_len=MAX_LEN):
    """Create and compile TensorFlow model on top of RoBERTa."""
    input_ids = keras.Input(shape=(max_len,), dtype='int32')
    att_masks = keras.Input(shape=(max_len,), dtype='int32')
    outputs = roberta_model(input_ids, attention_mask=att_masks).logits
    preds = keras.layers.Dense(3, activation='softmax')(outputs)
    
    model = keras.Model(inputs=[input_ids, att_masks], outputs=preds)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-5),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.CategoricalAccuracy()]
    )
    return model

def load_data(file_path, test_size=0.1):
    """Load CSV, preprocess, oversample, and split into train/val sets."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path, encoding='ISO-8859-1')
    print(f"Loaded {len(df)} samples")
    
    df = df.drop_duplicates(subset='Sentence')
    print(f"After removing duplicates: {len(df)} samples")
    
    df = preprocess_text(df, text_col='Sentence')

    X = df['text_clean'].values
    y = df['Sentiment'].values

    # Oversample to balance classes
    ros = RandomOverSampler(random_state=SEED)
    X_res, y_res = ros.fit_resample(X.reshape(-1, 1), y.reshape(-1, 1))
    X = X_res.flatten()
    y = y_res.flatten()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=SEED
    )

    # One-hot encode labels
    ohe = preprocessing.OneHotEncoder(sparse_output=False)
    y_train_ohe = ohe.fit_transform(y_train.reshape(-1, 1))
    y_val_ohe = ohe.transform(y_val.reshape(-1, 1))

    return (X_train, y_train_ohe), (X_val, y_val_ohe)

def evaluate_model(model, X_test, y_test_ohe):
    """Predict and print comprehensive classification metrics with visualizations."""
    test_ids, test_masks = tokenize_data(X_test)
    preds = model.predict([test_ids, test_masks])
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(y_test_ohe, axis=1)

    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS - FINANCIAL SENTIMENT")
    print("="*60)

    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    print(f"\nOverall Performance:")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted F1-Score:  {f1_weighted:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")

    # Detailed classification report
    class_names = ['negative', 'neutral', 'positive']
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Per-class metrics
    precision, recall, f1_per_class, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    print(f"\nPer-Class Metrics Summary:")
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
    print("-" * 50)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<10} {precision[i]:<10.3f} {recall[i]:<10.3f} {f1_per_class[i]:<10.3f} {support[i]:<10}")

    print("="*60)

    # Create visualizations
    create_evaluation_visualizations(y_true, y_pred, class_names)

def create_evaluation_visualizations(y_true, y_pred, class_names):
    """Create comprehensive evaluation visualizations."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Confusion Matrix - Raw Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix - Raw Counts')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # 2. Confusion Matrix - Normalized
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix - Normalized')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    # 3. Per-class accuracy
    per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
    bars = ax3.bar(class_names, per_class_accuracy, color=['#ff7f0e', '#2ca02c', '#1f77b4'], alpha=0.8)
    ax3.set_title('Per-Class Accuracy')
    ax3.set_ylabel('Accuracy')
    ax3.set_ylim(0, 1)
    
    # Annotate bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
    
    # 4. Precision, Recall, F1-Score comparison
    precision, recall, f1_per_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    x = np.arange(len(class_names))
    width = 0.25
    
    bars1 = ax4.bar(x - width, precision, width, label='Precision', color='#ff7f0e', alpha=0.8)
    bars2 = ax4.bar(x, recall, width, label='Recall', color='#2ca02c', alpha=0.8)
    bars3 = ax4.bar(x + width, f1_per_class, width, label='F1-Score', color='#1f77b4', alpha=0.8)
    
    ax4.set_title('Precision, Recall, and F1-Score by Class')
    ax4.set_ylabel('Score')
    ax4.set_xlabel('Class')
    ax4.set_xticks(x)
    ax4.set_xticklabels(class_names)
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()

def plot_sentiment_distribution(df, sentiment_col='Sentiment'):
    """Plot sentiment distribution using standardized style."""
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=sentiment_col, palette='Set2')
    plt.title('Count of Reviews by Sentiment (Financial Sentences)')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')

    # Annotate the bars with counts
    for p in plt.gca().patches:
        plt.gca().annotate(f"{int(p.get_height())}", 
                          (p.get_x() + p.get_width() / 2., p.get_height()), 
                          ha='center', va='bottom')
    plt.show()

def main(data_file=None):
    """Main function to run the financial sentiment analysis pipeline."""
    try:
        # Get dataset filepath
        if data_file is None:
            data_file = input("Enter path to your financial dataset CSV file (default: 'data.csv'): ").strip()
            if not data_file:
                data_file = 'data.csv'
        
        print(f"Using dataset: {data_file}")
        print("Loading and preprocessing data...")
        (X_train, y_train), (X_val, y_val) = load_data(data_file)
        
        # Optional: Generate data visualizations
        create_visualizations = input("Generate data visualizations? (y/n): ").lower() == 'y'
        if create_visualizations:
            try:
                # Load original data for visualization
                df = pd.read_csv(data_file, encoding='ISO-8859-1')
                df = preprocess_text(df, text_col='Sentence')
                print("\nGenerating data visualizations...")
                plot_sentiment_distribution(df)
            except Exception as e:
                print(f"Visualization error: {e}")
        
        # Tokenize data
        train_ids, train_masks = tokenize_data(X_train)
        val_ids, val_masks = tokenize_data(X_val)

        print(f"\nTraining samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")

        # Build and train model
        model = build_model()
        print("\nTraining model...")
        history = model.fit(
            [train_ids, train_masks], y_train, 
            validation_data=([val_ids, val_masks], y_val),
            epochs=3, batch_size=80, verbose=1
        )

        # Save model weights
        model.save_weights('RobertaWeightsFinancialSentiment.h5')
        print("Model weights saved!")

        # Evaluate model
        print("\nEvaluating model...")
        evaluate_model(model, X_val, y_val)
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        main(data_file)
    else:
        main()
