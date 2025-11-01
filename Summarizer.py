"""
Academic Note Summarizer - Google Colab Version
Run this entire notebook in Google Colab!

Instructions:
1. Go to https://colab.research.google.com
2. File → New Notebook
3. Copy and paste this code into cells
4. Run each cell in order
"""

# ============================================================
# CELL 1: Install Required Packages
# ============================================================
print("📦 Installing required packages...")
!pip install -q scikit-learn nltk

import nltk
print("📥 Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

print("✅ All packages installed successfully!")

# ============================================================
# CELL 2: Import Libraries
# ============================================================
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
import re
from collections import defaultdict
import json
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

print("✅ Libraries imported successfully!")

# ============================================================
# CELL 3: Define the ML Summarizer Class
# ============================================================
class MLNoteSummarizer:
    def __init__(self):
        self.vectorizer = None
        self.lsa_model = None
        self.training_data = []
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\[[\d,\s-]+\]', '', text)
        text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', text)
        text = re.sub(r'http[s]?://\S+', '', text)
        text = re.sub(r'[^\w\s.!?]', '', text)
        return text.strip()

    def train_model(self, documents, n_components=50):
        """Train the ML model"""
        print(f"🔧 Training model with {len(documents)} documents...")

        processed_docs = [self.preprocess_text(doc) for doc in documents]

        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.8,
            stop_words='english',
            ngram_range=(1, 2)
        )

        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)

        n_comp = min(n_components, tfidf_matrix.shape[1]-1, tfidf_matrix.shape[0]-1)
        self.lsa_model = TruncatedSVD(n_components=n_comp)
        lsa_matrix = self.lsa_model.fit_transform(tfidf_matrix)

        normalizer = Normalizer(copy=False)
        lsa_matrix = normalizer.fit_transform(lsa_matrix)

        self.training_data = documents

        variance_explained = sum(self.lsa_model.explained_variance_ratio_)
        print(f"✅ Model trained! Explained variance: {variance_explained:.2%}")

    def ml_summarize(self, text, num_sentences=3):
        """Generate summary using trained ML model"""
        if not self.vectorizer or not self.lsa_model:
            return "❌ Model not trained. Please train first."

        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        try:
            sentence_vectors = self.vectorizer.transform(sentences)
            sentence_scores = np.asarray(sentence_vectors.sum(axis=1)).ravel()

            top_indices = sentence_scores.argsort()[-num_sentences:][::-1]
            top_indices = sorted(top_indices)

            return ' '.join([sentences[i] for i in top_indices])
        except:
            return self.extractive_summarize(text, num_sentences)

    def extractive_summarize(self, text, num_sentences=3):
        """Traditional extractive summarization"""
        sentences = sent_tokenize(text)

        if len(sentences) <= num_sentences:
            return text

        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)

        for word in words:
            if word not in self.stop_words and word.isalnum() and len(word) > 2:
                word_freq[word] += 1

        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words_in_sentence)
            if len(words_in_sentence) > 0:
                sentence_scores[i] = score / len(words_in_sentence)

        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)

        return ' '.join([sentences[i] for i in top_indices])

    def extract_keywords(self, text, top_n=10):
        """Extract keywords"""
        if self.vectorizer:
            try:
                tfidf = self.vectorizer.transform([self.preprocess_text(text)])
                feature_names = self.vectorizer.get_feature_names_out()
                scores = tfidf.toarray()[0]
                top_indices = scores.argsort()[-top_n:][::-1]
                return [feature_names[i] for i in top_indices if scores[i] > 0]
            except:
                pass

        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)

        for word in words:
            if word not in self.stop_words and word.isalnum() and len(word) > 3:
                word_freq[word] += 1

        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:top_n]]

    def generate_bullet_points(self, text, num_points=5):
        """Generate bullet points"""
        sentences = sent_tokenize(text)

        indicators = ['important', 'significant', 'key', 'main', 'crucial',
                     'essential', 'result', 'finding', 'conclude', 'demonstrate']

        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            words = word_tokenize(sentence.lower())

            for indicator in indicators:
                if indicator in words:
                    score += 2

            if i < len(sentences) * 0.2:
                score += 1
            if i > len(sentences) * 0.8:
                score += 1

            if 10 <= len(words) <= 30:
                score += 1

            scored_sentences.append((sentence, score))

        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, score in scored_sentences[:num_points]]

print("✅ MLNoteSummarizer class defined!")

# ============================================================
# CELL 4: Training Data (Academic Documents)
# ============================================================
training_documents = [
    """Machine learning is a method of data analysis that automates analytical model building.
    It is a branch of artificial intelligence based on the idea that systems can learn from data,
    identify patterns and make decisions with minimal human intervention. The iterative aspect of
    machine learning is important because as models are exposed to new data, they are able to
    independently adapt.""",

    """Neural networks are computing systems inspired by biological neural networks.
    Such systems learn to perform tasks by considering examples, generally without being programmed
    with task-specific rules. A neural network is based on a collection of connected units called
    artificial neurons. Deep learning uses multiple layers to progressively extract higher-level
    features from raw input.""",

    """Natural language processing is a subfield of linguistics, computer science, and artificial
    intelligence concerned with the interactions between computers and human language. NLP is used
    to apply machine learning algorithms to text and speech. Common applications include sentiment
    analysis, machine translation, and question answering systems.""",

    """Data science is an interdisciplinary field that uses scientific methods to extract knowledge
    from structured and unstructured data. Data science combines aspects of statistics, computer
    science, and domain expertise. Important techniques include data mining, machine learning,
    and predictive analytics.""",

    """Computer vision is an interdisciplinary field that deals with how computers can gain
    high-level understanding from digital images or videos. Applications include object detection,
    facial recognition, and autonomous vehicles. Deep learning has revolutionized computer vision
    in recent years.""",

    """Artificial intelligence encompasses machine learning, deep learning, and neural networks.
    AI systems can now perform tasks that typically require human intelligence. The field has
    applications in healthcare, finance, transportation, and entertainment. Ethical considerations
    around AI include bias, privacy, and accountability.""",
]

print(f"✅ Loaded {len(training_documents)} training documents")

# ============================================================
# CELL 5: Initialize and Train the Model
# ============================================================
print("="*60)
print("🚀 INITIALIZING ACADEMIC NOTE SUMMARIZER")
print("="*60)

summarizer = MLNoteSummarizer()
summarizer.train_model(training_documents, n_components=50)

print("\n✅ Model ready for use!")

# ============================================================
# CELL 6: Example Test - Summarize Sample Text
# ============================================================
sample_text = """
Artificial intelligence has become one of the most transformative technologies of the 21st century.
Machine learning, a subset of AI, enables computers to learn and improve from experience without
being explicitly programmed. Deep learning, which uses neural networks with multiple layers, has
achieved remarkable success in areas like image recognition and natural language processing.

The history of AI dates back to the 1950s when Alan Turing proposed the famous Turing Test.
However, practical applications remained limited until recent decades due to computational constraints.
The availability of big data and powerful GPUs has enabled the current AI revolution. Major tech
companies are investing billions in AI research and development.

Key applications of AI include autonomous vehicles, medical diagnosis, financial trading, and
personal assistants. In healthcare, AI algorithms can detect diseases from medical images with
accuracy comparable to human experts. In finance, AI powers fraud detection and algorithmic trading.
The technology also raises important ethical questions about privacy, bias, and job displacement.

Looking forward, AI is expected to continue advancing rapidly. Researchers are working on artificial
general intelligence that can perform any intellectual task a human can do. However, many experts
emphasize the importance of developing AI responsibly with proper safeguards and ethical guidelines.
"""

print("="*60)
print("📝 EXAMPLE SUMMARIZATION")
print("="*60)

print("\n🤖 ML-BASED SUMMARY:")
print("-"*60)
print(summarizer.ml_summarize(sample_text, num_sentences=3))

print("\n\n📄 EXTRACTIVE SUMMARY:")
print("-"*60)
print(summarizer.extractive_summarize(sample_text, num_sentences=3))

print("\n\n🔑 KEY TERMS:")
print("-"*60)
keywords = summarizer.extract_keywords(sample_text, top_n=10)
print(", ".join(keywords))

print("\n\n📌 BULLET POINTS:")
print("-"*60)
bullet_points = summarizer.generate_bullet_points(sample_text, num_points=5)
for i, point in enumerate(bullet_points, 1):
    print(f"{i}. {point}")

# ============================================================
# CELL 7: Interactive Summarization
# ============================================================
def summarize_notes(your_text, method='both'):
    """
    Summarize your notes

    Args:
        your_text: Your academic notes as a string
        method: 'ml', 'extractive', or 'both'
    """
    print("\n" + "="*60)
    print("📚 SUMMARIZING YOUR NOTES")
    print("="*60)

    if method in ['ml', 'both']:
        print("\n🤖 ML-BASED SUMMARY:")
        print("-"*60)
        print(summarizer.ml_summarize(your_text, num_sentences=3))

    if method in ['extractive', 'both']:
        print("\n\n📄 EXTRACTIVE SUMMARY:")
        print("-"*60)
        print(summarizer.extractive_summarize(your_text, num_sentences=3))

    print("\n\n🔑 KEY TERMS:")
    print("-"*60)
    keywords = summarizer.extract_keywords(your_text, top_n=10)
    print(", ".join(keywords))

    print("\n\n📌 BULLET POINTS:")
    print("-"*60)
    bullet_points = summarizer.generate_bullet_points(your_text, num_points=5)
    for i, point in enumerate(bullet_points, 1):
        print(f"{i}. {point}")

    print("\n" + "="*60)
    print(f"📊 Statistics: {len(your_text.split())} words")
    print("="*60)

print("✅ Interactive function ready!")
print("\nTo use, run:")
print("summarize_notes(your_text_here)")

# ============================================================
# CELL 8: Quick Usage Examples
# ============================================================
print("\n" + "="*60)
print("💡 HOW TO USE IN COLAB")
print("="*60)
print("""
# Method 1: Paste your text directly
my_notes = '''
Your academic notes here...
Can be multiple paragraphs...
'''
summarize_notes(my_notes)

# Method 2: Upload a text file
from google.colab import files
uploaded = files.upload()
filename = list(uploaded.keys())[0]
with open(filename, 'r') as f:
    text = f.read()
summarize_notes(text)

# Method 3: Train with your own documents
my_training_data = [
    "Document 1 text...",
    "Document 2 text...",
    "Document 3 text..."
]
summarizer.train_model(my_training_data)
summarize_notes(my_notes)
""")

# ============================================================
# CELL 9: File Upload Feature
# ============================================================
def upload_and_summarize():
    """Upload a text file and summarize it"""
    from google.colab import files

    print("📤 Upload your text file (.txt):")
    uploaded = files.upload()

    if not uploaded:
        print("❌ No file uploaded")
        return

    filename = list(uploaded.keys())[0]

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"\n✅ File '{filename}' loaded successfully!")
        print(f"📊 File size: {len(text)} characters, {len(text.split())} words\n")

        summarize_notes(text)
    except Exception as e:
        print(f"❌ Error reading file: {e}")

print("✅ File upload function ready!")
print("\nTo upload and summarize a file, run:")
print("upload_and_summarize()")

# ============================================================
# CELL 10: Custom Training
# ============================================================
def train_with_my_documents(documents_list):
    """
    Train the model with your own academic documents

    Args:
        documents_list: List of strings, each string is a document
    """
    print(f"\n🔧 Training with {len(documents_list)} new documents...")
    summarizer.train_model(documents_list, n_components=50)
    print("✅ Model retrained successfully!")

print("✅ Custom training function ready!")
print("\nTo train with your documents:")
print("""
my_docs = [
    "Your first document...",
    "Your second document...",
    "Your third document..."
]
train_with_my_documents(my_docs)
""")

# ============================================================
# READY TO USE!
# ============================================================
print("\n" + "="*60)
print("🎉 EVERYTHING IS READY!")
print("="*60)
print("""
Quick Start:
1. Paste your notes into a variable
2. Run: summarize_notes(your_notes)

Example:
---------
my_lecture_notes = '''
Your lecture content here...
'''
summarize_notes(my_lecture_notes)

Or upload a file:
-----------------
upload_and_summarize()
""")
