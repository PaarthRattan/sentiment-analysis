#!/usr/bin/env python3
"""
VADER Twitter Sentiment Analysis

This module implements VADER (Valence Aware Dictionary for Sentiment Reasoning)
sentiment analysis on Twitter data, providing comprehensive evaluation metrics
for comparison with transformer-based models like RoBERTa.

Author: Paarth Rattan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, 
    recall_score, classification_report, confusion_matrix
)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('ggplot')
sns.set_palette("husl")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class TwitterSentimentVADER:
    """
    A class for performing VADER sentiment analysis on Twitter data.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.data = None
        self.min_words = 4
    
    def clean_text(self, text):
        """Clean text by removing URLs, special characters, and extra spaces."""
        if pd.isna(text):
            return ""
        
        text = str(text)
        # Remove URLs and web links
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub(r'\s+', ' ', text, flags=re.I)
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('<.*?>+', '', text)
        return text
    
    def clean_hashtags(self, text):
        """Clean hashtags and keep meaningful content."""
        # Remove last hashtags
        new_text = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', text))
        # Remove hashtag symbols from words in the middle
        new_text2 = " ".join(word.strip() for word in re.split('#|_', new_text))
        return new_text2
    
    def remove_multiple_spaces(self, text):
        """Remove multiple consecutive spaces."""
        return re.sub(r"\s{2,}", " ", text).strip()
    
    def count_words(self, text):
        """Count words in text."""
        return len(text.split()) if text else 0
    
    def load_and_preprocess_data(self, file_path, sample_size=5000):
        """Load and preprocess the Twitter dataset."""
        try:
            # Load data
            self.data = pd.read_csv(file_path, encoding='ISO-8859-1')
            print(f"Loaded {len(self.data)} samples")
            
            # Drop unnecessary columns if they exist
            cols_to_drop = ['textID', 'selected_text']
            self.data = self.data.drop([col for col in cols_to_drop if col in self.data.columns], axis=1)
            
            # Sample data if specified
            if sample_size and len(self.data) > sample_size:
                self.data = self.data.head(sample_size)
                print(f"Using sample of {sample_size} tweets")
            
            # Handle missing values
            self.data = self.data.replace(r'^\s*$', np.nan, regex=True)
            self.data = self.data.dropna()
            
            # Clean text
            print("Cleaning text data...")
            cleaned_texts = []
            for text in tqdm(self.data['text'], desc="Processing tweets"):
                if pd.notna(text):
                    cleaned = self.remove_multiple_spaces(
                        self.clean_hashtags(
                            self.clean_text(str(text))
                        )
                    )
                    cleaned_texts.append(cleaned)
                else:
                    cleaned_texts.append("")
            
            self.data['text_cleaned'] = cleaned_texts
            
            # Calculate word count and filter
            self.data['text_len'] = self.data['text_cleaned'].apply(self.count_words)
            initial_count = len(self.data)
            self.data = self.data[self.data['text_len'] > self.min_words].reset_index(drop=True)
            
            print(f"Filtered data: {len(self.data)} samples (removed {initial_count - len(self.data)} samples with ≤{self.min_words} words)")
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def calculate_sentiment(self, text):
        """
        Calculate sentiment using VADER.
        
        Returns:
            int: -1 for negative, 0 for neutral, 1 for positive
        """
        if not text or pd.isna(text):
            return 0
        
        sentiment_scores = self.analyzer.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        
        if compound_score >= 0.1:
            return 1  # Positive
        elif compound_score <= -0.1:
            return -1  # Negative
        else:
            return 0  # Neutral
    
    def analyze_sentiments(self):
        """Analyze sentiments for all texts in the dataset."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        print("Analyzing sentiments with VADER...")
        self.data['vader_sentiment'] = [
            self.calculate_sentiment(text) 
            for text in tqdm(self.data['text_cleaned'], desc="Processing")
        ]
        
        return self.data
    
    def map_true_labels(self):
        """Map true labels to numerical format."""
        label_mapping = {'negative': -1, 'neutral': 0, 'positive': 1}
        self.data['true_sentiment_num'] = self.data['sentiment'].map(label_mapping)
        return self.data
    
    def calculate_comprehensive_metrics(self):
        """Calculate comprehensive evaluation metrics."""
        if 'vader_sentiment' not in self.data.columns or 'true_sentiment_num' not in self.data.columns:
            print("Missing required columns. Run analysis first.")
            return None
        
        y_true = self.data['true_sentiment_num']
        y_pred = self.data['vader_sentiment']
        
        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'precision_macro': precision_score(y_true, y_pred, average='macro'),
            'recall_macro': recall_score(y_true, y_pred, average='macro')
        }
        
        # Per-class metrics
        class_names = ['negative', 'neutral', 'positive']
        class_labels = [-1, 0, 1]
        
        per_class_metrics = {}
        for label, name in zip(class_labels, class_names):
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            per_class_metrics[name] = {
                'precision': precision_score(y_true_binary, y_pred_binary, average='binary', zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, average='binary', zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, average='binary', zero_division=0),
                'support': (y_true == label).sum()
            }
        
        return overall_metrics, per_class_metrics
    
    def print_evaluation_results(self):
        """Print comprehensive evaluation results."""
        overall_metrics, per_class_metrics = self.calculate_comprehensive_metrics()
        
        if overall_metrics is None:
            return
        
        print("\n" + "="*60)
        print("VADER TWITTER SENTIMENT ANALYSIS - EVALUATION RESULTS")
        print("="*60)
        
        # Overall performance
        print(f"\nOverall Performance:")
        print(f"Accuracy:           {overall_metrics['accuracy']:.4f}")
        print(f"F1-Score (Macro):   {overall_metrics['f1_macro']:.4f}")
        print(f"F1-Score (Weighted): {overall_metrics['f1_weighted']:.4f}")
        print(f"Precision (Macro):  {overall_metrics['precision_macro']:.4f}")
        print(f"Recall (Macro):     {overall_metrics['recall_macro']:.4f}")
        
        # Per-class performance
        print(f"\nPer-Class Performance:")
        print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 50)
        for class_name, metrics in per_class_metrics.items():
            print(f"{class_name:<10} {metrics['precision']:<10.3f} {metrics['recall']:<10.3f} "
                  f"{metrics['f1']:<10.3f} {metrics['support']:<10}")
        
        # Classification report
        y_true = self.data['true_sentiment_num']
        y_pred = self.data['vader_sentiment']
        class_names = ['negative', 'neutral', 'positive']
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
        
        print("="*60)
        
        return overall_metrics, per_class_metrics
    
    def plot_sentiment_distribution(self):
        """Plot sentiment distribution using your preferred style."""
        plt.figure(figsize=(8, 5))
        sns.countplot(data=self.data, x='sentiment', palette='Set2')
        plt.title('Count of Reviews by Sentiment (Twitter tweets)')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')

        # Annotate the bars with counts
        for p in plt.gca().patches:
            plt.gca().annotate(f"{int(p.get_height())}", 
                             (p.get_x() + p.get_width() / 2., p.get_height()), 
                             ha='center', va='bottom')
        plt.show()
    
    def plot_sentiment_comparison(self):
        """Plot comparison between true and predicted sentiments."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # True sentiment distribution
        true_counts = self.data['sentiment'].value_counts()
        bars1 = ax1.bar(true_counts.index, true_counts.values, color=['#ff6b6b', '#feca57', '#48dbfb'], alpha=0.8)
        ax1.set_title('True Sentiment Distribution')
        ax1.set_xlabel('Sentiment')
        ax1.set_ylabel('Count')
        
        # Add count labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        # VADER predicted sentiment distribution
        vader_mapping = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        vader_labels = [vader_mapping[pred] for pred in self.data['vader_sentiment']]
        vader_counts = pd.Series(vader_labels).value_counts()
        
        bars2 = ax2.bar(vader_counts.index, vader_counts.values, color=['#ff6b6b', '#feca57', '#48dbfb'], alpha=0.8)
        ax2.set_title('VADER Predicted Sentiment Distribution')
        ax2.set_xlabel('Sentiment')
        ax2.set_ylabel('Count')
        
        # Add count labels
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{int(height)}', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
    
    def plot_word_length_distribution(self):
        """Plot word length distribution using your preferred style."""
        plt.figure(figsize=(7,5))
        ax = sns.countplot(x='text_len', data=self.data[self.data['text_len']<10], palette='mako')
        plt.title('Twitter tweets with less than 10 words')
        plt.yticks([])
        ax.bar_label(ax.containers[0])
        plt.ylabel('count')
        plt.xlabel('Word Count')
        plt.show()
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix with your preferred style."""
        y_true = self.data['true_sentiment_num']
        y_pred = self.data['vader_sentiment']
        
        cm = confusion_matrix(y_true, y_pred)
        class_names = ['Negative', 'Neutral', 'Positive']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix - Raw Counts')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Normalized
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_norm, annot=True, fmt='.3f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix - Normalized')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_accuracy_by_category(self):
        """Plot accuracy by sentiment category using your preferred style."""
        y_true = self.data['true_sentiment_num']
        y_pred = self.data['vader_sentiment']
        
        # Calculate per-class accuracy
        class_names = ['negative', 'neutral', 'positive']
        class_labels = [-1, 0, 1]
        accuracies = []
        
        for label in class_labels:
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            accuracies.append(accuracy)
        
        # Create bar plot
        plt.figure(figsize=(8, 6))
        bars = plt.bar(class_names, accuracies, color=['red', 'blue', 'green'], alpha=0.8)
        plt.xlabel('Sentiment Category')
        plt.ylabel('Accuracy')
        plt.title('VADER Accuracy by Sentiment Category')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, accuracy + 0.01, 
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        plt.show()

def main(data_file=None, sample_size=5000):
    """Main function to run VADER Twitter sentiment analysis."""
    # Get dataset filepath
    if data_file is None:
        data_file = input("Enter path to your Twitter dataset CSV file (default: 'Tweets.csv'): ").strip()
        if not data_file:
            data_file = 'Tweets.csv'
    
    print(f"Using dataset: {data_file}")
    
    # Initialize VADER analyzer
    vader_analyzer = TwitterSentimentVADER()
    
    # Load and preprocess data
    data = vader_analyzer.load_and_preprocess_data(data_file, sample_size)
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Map true labels
    vader_analyzer.map_true_labels()
    
    # Analyze sentiments
    vader_analyzer.analyze_sentiments()
    
    # Generate visualizations
    create_visualizations = input("Generate visualizations? (y/n): ").lower() == 'y'
    if create_visualizations:
        print("\nGenerating visualizations...")
        vader_analyzer.plot_sentiment_distribution()
        vader_analyzer.plot_word_length_distribution()
        vader_analyzer.plot_sentiment_comparison()
        vader_analyzer.plot_confusion_matrix()
        vader_analyzer.plot_accuracy_by_category()
    
    # Print evaluation results
    print("\nEvaluating VADER performance on Twitter data...")
    vader_analyzer.print_evaluation_results()
    
    return vader_analyzer

if __name__ == "__main__":
    import sys
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
        sample_size = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
        main(data_file, sample_size)
    else:
        main()
