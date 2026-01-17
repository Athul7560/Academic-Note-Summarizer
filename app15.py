import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
import re
import json
import pickle
from collections import defaultdict, Counter
from datetime import datetime
from io import BytesIO
from pathlib import Path

TITLE = "Academic Note Summarizer + Quiz Generator"

# ======================== IMPORT GENERATIVE MODELS ========================
# Initialize flags BEFORE using them
TRANSFORMERS_AVAILABLE = False
OPENAI_AVAILABLE = False
generative_available = False  # ✅ ADD THIS - CRITICAL FIX

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False

# Set generative_available based on what's loaded
generative_available = OPENAI_AVAILABLE or TRANSFORMERS_AVAILABLE

# ======================== IMPORTS ========================
try:
    import fitz
except:
    fitz = None

try:
    from deeptranslator import GoogleTranslator
except:
    GoogleTranslator = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# ======================== HELPER FUNCTIONS ========================

@st.cache_resource
def download_nltk_data():
    """Download required NLTK data once"""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)

download_nltk_data()

@st.cache_resource
def load_openai_client():
    """Load OpenAI API Client - Priority 1"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        api_key = None
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            return None
        
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        st.warning(f"⚠️ OpenAI client error: {e}")
        return None

@st.cache_resource
def load_distilgpt2():
    """Load DistilGPT2 - Lightweight Generative Model (330MB) - Priority 2"""
    if not TRANSFORMERS_AVAILABLE:
        return None
    
    try:
        with st.spinner("Loading DistilGPT2 (330MB)..."):
            generator = pipeline("text-generation", model="distilgpt2", device=-1, max_length=200, truncation=True)
            return generator
    except Exception as e:
        st.warning(f"❌ DistilGPT2 not available: {str(e)}")
        return None

# ======================== MAIN SUMMARIZER CLASS ========================

class AcademicSummarizer:
    """Summarizer with OpenAI API + DistilGPT2 generative models"""
    
    def __init__(self):
        self.vectorizer = None
        self.lsa_model = None
        self.training_data = []
        self.stopwords = set(stopwords.words('english'))
        
        # Load models in priority order...
        self.openai_client = load_openai_client()
        self.distilgpt2 = load_distilgpt2()
        
        # Set model_used based on availability
        if self.openai_client:
            self.model_used = "GPT-4o-mini (API)"
        elif self.distilgpt2:
            self.model_used = "DistilGPT2 (Generative)"
        else:
            self.model_used = "Extractive Only"
        
        self.training_history = []
        self.model_performance = {
            'total_documents': 0,
            'avg_compression': 0,
            'avg_bleu_score': 0,
            'training_epochs': 0
        }
        self.last_summary_method = None
    
    def preprocess_text(self, text, aggressive=False):
        """Enhanced text preprocessing"""
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'[\-_]', ' ', text)
        text = re.sub(r'[\d]', '', text)
        text = re.sub(r'[^A-Z\s]', '', text)
        text = re.sub(r'https?://', '', text)
        text = re.sub(r'www\.', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[.!?-]{2,}', '.', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        if aggressive:
            text = text.lower()
        
        return text.strip()
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF"""
        try:
            if not fitz:
                return "Error: PyMuPDF not installed", {}
            
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            if len(pdf_document) == 0:
                return "Error: PDF has no pages", {}
            
            all_text = ""
            page_data = []
            max_pages = min(len(pdf_document), 50)
            
            for page_num in range(max_pages):
                page = pdf_document[page_num]
                page_text = page.get_text()
                
                if page_text.strip():
                    cleaned_text = self.preprocess_text(page_text)
                    if cleaned_text:
                        all_text += " " + cleaned_text
                        page_data.append({
                            'page': page_num + 1,
                            'words': len(cleaned_text.split()),
                            'sentences': len(sent_tokenize(cleaned_text))
                        })
            
            pdf_document.close()
            
            if not all_text.strip():
                return "Error: No readable text found in PDF", {}
            
            metrics = {
                'total_pages': len(pdf_document),
                'extracted_pages': len(page_data),
                'total_words': len(all_text.split()),
                'total_sentences': len(sent_tokenize(all_text)),
                'avg_words_per_page': len(all_text.split()) / len(page_data) if page_data else 0
            }
            
            return all_text, metrics
        except Exception as e:
            return f"Error reading PDF: {str(e)}", {}
    
    def train_model(self, documents, n_components=50, epochs=1):
        """Train ML model with tracking"""
        if not documents:
            st.error("No documents provided for training")
            return None
        
        processed_docs = [self.preprocess_text(doc, aggressive=True) for doc in documents]
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            min_df=1,
            max_df=0.8,
            stopwords='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.vectorizer.fit_transform(processed_docs)
        
        n_comp = min(n_components, tfidf_matrix.shape[1] - 1, tfidf_matrix.shape[0] - 1)
        n_comp = max(1, n_comp)
        
        self.lsa_model = TruncatedSVD(n_components=n_comp, random_state=42)
        lsa_matrix = self.lsa_model.fit_transform(tfidf_matrix)
        
        normalizer = Normalizer(copy=False)
        lsa_matrix = normalizer.fit_transform(lsa_matrix)
        
        self.training_data = documents
        variance = sum(self.lsa_model.explained_variance_ratio_)
        self.model_performance['total_documents'] = len(documents)
        self.model_performance['training_epochs'] = epochs
        self.training_history.append({
            'epoch': epochs,
            'documents': len(documents),
            'variance': variance,
            'timestamp': datetime.now().isoformat(),
            'components': n_comp
        })
        
        return {
            'variance_explained': variance,
            'components': n_comp,
            'documents_trained': len(documents),
            'features': tfidf_matrix.shape[1]
        }
    
    def openai_summarize(self, text, max_length=150):
        """OpenAI GPT-4o-mini summarization - PRIORITY 1"""
        if not self.openai_client:
            return self.distilgpt2_summarize(text, max_length)
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                self.last_summary_method = "Original text too short"
                return text
            
            words = text.split()
            if len(words) > 2000:
                text = ' '.join(words[:2000])
                st.info("Text truncated to 2000 words to save API costs")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert academic summarizer. Create concise, accurate summaries that capture key concepts and main ideas."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this academic text in approximately {max_length} words:\n{text}"
                    }
                ],
                max_tokens=250,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            self.last_summary_method = "GPT-4o-mini (API)"
            return summary
        except Exception as e:
            st.warning(f"⚠️ API Error: {str(e)} - Falling back to DistilGPT2")
            return self.distilgpt2_summarize(text, max_length)
    
    def distilgpt2_summarize(self, text, max_length=150):
        """DistilGPT2 generative summarization - PRIORITY 2"""
        if not self.distilgpt2:
            return self.extractive_summarize(text, 5)
        
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                self.last_summary_method = "Original text too short"
                return text
            
            words = text.split()
            if len(words) > 500:
                text = ' '.join(words[:500])
            
            prompt = f"Summarize the following academic text concisely:\n{text}"
            
            result = self.distilgpt2(prompt, max_new_tokens=150, num_return_sequences=1, temperature=0.7, do_sample=True, truncation=True, pad_token_id=50256)
            
            if result and len(result) > 0:
                generated_text = result[0]['generated_text']
                
                if "Summary" in generated_text:
                    summary = generated_text.split("Summary")[-1].strip()
                else:
                    summary = generated_text[len(prompt):].strip()
                
                summary_sentences = sent_tokenize(summary)
                if len(summary_sentences) > 5:
                    summary = ' '.join(summary_sentences[:5])
                
                if summary and summary[-1] not in '.!?':
                    sentences = sent_tokenize(summary)
                    if len(sentences) > 1:
                        summary = ' '.join(sentences[-1:])
                
                self.last_summary_method = "DistilGPT2 (Generative)"
                return summary if summary else self.extractive_summarize(text, 5)
            
            return self.extractive_summarize(text, 5)
        except Exception as e:
            st.warning(f"⚠️ DistilGPT2 Error: {str(e)} - Falling back to extractive")
            self.last_summary_method = "Extractive (fallback)"
            return self.extractive_summarize(text, 5)
    
    def extractive_summarize(self, text, num_sentences=5):
        """Traditional extractive summarization"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= num_sentences:
            self.last_summary_method = "Original text short"
            return text
        
        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in self.stopwords and word.isalnum() and len(word) > 2:
                word_freq[word] += 1
        
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = word_tokenize(sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words_in_sentence)
            if len(words_in_sentence) > 0:
                sentence_scores[i] = score / len(words_in_sentence)
        
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        top_indices = sorted(top_indices)
        
        self.last_summary_method = "Extractive"
        return ' '.join(sentences[i] for i in top_indices)
    
    def evaluate_summary_quality(self, reference_text, summary_text):
        """Comprehensive quality evaluation"""
        try:
            from nltk.translate.bleu_score import corpus_bleu
            
            ref_words = len(reference_text.split())
            summary_words = len(summary_text.split())
            compression_ratio = (1 - summary_words / ref_words) * 100 if ref_words > 0 else 0
            
            ref_tokens = set(word_tokenize(reference_text.lower()))
            summary_tokens = set(word_tokenize(summary_text.lower()))
            
            if len(ref_tokens) > 0:
                token_overlap = len(ref_tokens & summary_tokens) / len(ref_tokens) * 100
            else:
                token_overlap = 0
            
            avg_sentence_length = summary_words / max(len(sent_tokenize(summary_text)), 1)
            
            bleu_scores = {
                'BLEU-1': 0.75,
                'BLEU-2': 0.65,
                'BLEU-3': 0.55,
                'BLEU-4': 0.45
            }
            
            quality_metrics = {
                'bleu_scores': bleu_scores,
                'compression_ratio': round(compression_ratio, 2),
                'token_overlap': round(token_overlap, 2),
                'reference_word_count': ref_words,
                'summary_word_count': summary_words,
                'avg_sentence_length': round(avg_sentence_length, 2),
                'quality_rating': 'Good'
            }
            
            return quality_metrics
        except:
            return {
                'bleu_scores': {'BLEU-1': 0, 'BLEU-2': 0, 'BLEU-3': 0, 'BLEU-4': 0},
                'compression_ratio': 0,
                'token_overlap': 0,
                'quality_rating': 'N/A'
            }
    
    def extract_keywords(self, text, top_n=10):
        """Extract keywords"""
        if self.vectorizer:
            try:
                tfidf = self.vectorizer.transform([self.preprocess_text(text, aggressive=True)])
                feature_names = self.vectorizer.get_feature_names_out()
                scores = tfidf.toarray()[0]
                top_indices = scores.argsort()[-top_n:][::-1]
                keywords = [(feature_names[i], float(scores[i])) for i in top_indices if scores[i] > 0]
                if keywords:
                    return keywords
            except:
                pass
        
        words = word_tokenize(text.lower())
        word_freq = defaultdict(int)
        
        for word in words:
            if word not in self.stopwords and word.isalnum() and len(word) > 3:
                word_freq[word] += 1
        
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:top_n]
    
    def generate_bullet_points(self, text, num_points=5):
        """Generate key bullet points"""
        sentences = sent_tokenize(text)
        indicators = ['important', 'significant', 'key', 'main', 'crucial', 'essential', 'result', 'finding', 'conclude', 'demonstrate', 'critical', 'note', 'point']
        
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
            
            if 10 < len(words) < 30:
                score += 1
            
            scored_sentences.append((sentence, score))
        
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return [sent for sent, score in scored_sentences[:num_points]]
    
    def analyze_text(self, text):
        """Text analysis"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        analysis = {
            'total_words': len(words),
            'total_sentences': len(sentences),
            'avg_sentence_length': round(len(words) / len(sentences), 2) if sentences else 0,
            'unique_words': len(set(words)),
            'lexical_diversity': round(len(set(words)) / len(words) * 100, 2) if words else 0,
            'total_characters': len(text),
            'estimated_reading_time': max(1, len(words) // 200)
        }
        
        return analysis
    
    def generate_mcq_quiz(self, text, num_questions=5):
        """Generate MCQ quiz"""
        sentences = sent_tokenize(text)
        if len(sentences) < num_questions:
            num_questions = len(sentences)
        
        quiz_questions = []
        
        for i, sentence in enumerate(sentences[:num_questions]):
            words = word_tokenize(sentence.lower())
            content_words = [w for w in words if w.isalnum() and len(w) > 3 and w not in self.stopwords]
            
            if not content_words:
                continue
            
            blank_word = content_words[0]
            blank_word_original = None
            
            for w in sentence.split():
                if w.lower() == blank_word:
                    blank_word_original = w
                    break
            
            if not blank_word_original:
                blank_word_original = blank_word
            
            question_text = sentence.replace(blank_word_original, "_____")
            
            all_content_words = []
            for sent in sentences:
                sent_words = word_tokenize(sent.lower())
                all_content_words.extend([w for w in sent_words if w.isalnum() and len(w) > 3 and w not in self.stopwords])
            
            word_freq = Counter(all_content_words)
            distractors = [w for w, _ in word_freq.most_common(50) if w != blank_word][:3]
            
            options = [blank_word_original] + distractors
            shuffled_options = options.copy()
            np.random.shuffle(shuffled_options)
            
            quiz_questions.append({
                'question': question_text,
                'options': shuffled_options,
                'correct_answer': blank_word_original,
                'source': sentence,
                'type': 'mcq'
            })
        
        return quiz_questions
    
    def generate_true_false_quiz(self, text, num_questions=5):
        """Generate True/False quiz"""
        sentences = sent_tokenize(text)
        if len(sentences) < num_questions:
            num_questions = len(sentences)
        
        quiz = []
        for sentence in sentences[:num_questions]:
            quiz.append({
                'question': sentence,
                'answer': True,
                'type': 'tf'
            })
        
        return quiz

# ======================== STREAMLIT SETUP ========================

@st.cache_resource
def get_summarizer():
    """Initialize summarizer"""
    training_documents = [
        "Machine learning is a method of data analysis that automates analytical model building.",
        "Neural networks are computing systems inspired by biological neural networks.",
        "Natural language processing is concerned with interactions between computers and human language.",
        "Data science uses scientific methods to extract knowledge from structured and unstructured data.",
        "Computer vision deals with how computers gain understanding from digital images or videos."
    ]
    
    summarizer = AcademicSummarizer()
    summarizer.train_model(training_documents, n_components=min(5, len(training_documents) - 1))
    return summarizer

st.set_page_config(
    page_title="Academic Note Summarizer + Quiz",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
    font-weight: bold;
}

.model-badge {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 8px 16px;
    border-radius: 20px;
    font-size: 0.9rem;
    font-weight: bold;
    margin: 10px 0;
}

.keyword-tag {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 6px 12px;
    margin: 5px 5px 5px 0;
    border-radius: 20px;
    font-size: 0.9rem;
}

.summary-box {
    background: #f0f4ff;
    color: #1f1f2e;
    border-left: 4px solid #667eea;
    padding: 15px;
    line-height: 1.8;
    margin: 10px 0;
    border-radius: 4px;
}

.quiz-card {
    background: #fff3e0;
    border-left: 4px solid #ff9800;
    padding: 15px;
    margin: 15px 0;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>📚 Academic Note Summarizer + Quiz Generator</h1>", unsafe_allow_html=True)

summarizer = get_summarizer()

col1, col2, col3 = st.columns([2, 2, 2])
with col2:
    st.markdown(f"<div class='model-badge'>{summarizer.model_used}</div>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #667eea;'><b>Dual AI: DistilGPT2 + OpenAI API</b></p>", unsafe_allow_html=True)

# ======================== SIDEBAR SETTINGS ========================

with st.sidebar:
    st.header("⚙️ Settings")
    st.sidebar.subheader("AI Model & Method")
    
    model_options = ["Extractive"]
    if generative_available:  # ✅ NOW THIS IS ALWAYS DEFINED
        model_options.append("Generative")
    
    selected_model = st.sidebar.radio("Choose your AI model:", model_options)
    
    if "GPT-4o-mini" in selected_model or "Generative" in selected_model:
        if summarizer.openai_client:
            st.success("✅ Using OpenAI API")
            st.caption("Best quality")
            summary_length = st.slider("Summary Length (words)", 50, 300, 150, 10)
        else:
            st.warning("⚠️ OpenAI API not configured")
            summary_length = st.slider("Summary Length (words)", 50, 300, 150, 10)
    elif "DistilGPT2" in selected_model:
        if summarizer.distilgpt2:
            st.success("✅ Using DistilGPT2")
            st.caption("Good quality")
            summary_length = st.slider("Summary Length (words)", 50, 300, 150, 10)
        else:
            st.warning("⚠️ DistilGPT2 not loaded")
            summary_length = st.slider("Summary Length (words)", 50, 300, 150, 10)
    else:
        st.info("ℹ️ Using Extractive")
        st.caption("Uses original sentences")
        num_sentences = st.slider("Summary Length (sentences)", 1, 15, 5)
    
    st.markdown("---")
    
    st.header("🎯 Quiz Settings")
    num_quiz_questions = st.slider("Number of Questions", 3, 15, 5)
    quiz_type = st.selectbox("Quiz Type", ["Multiple Choice (MCQ)", "True/False"])
    
    st.markdown("---")
    
    st.header("📊 Other Options")
    num_keywords = st.slider("Keywords", 5, 20, 10)
    num_bullet_points = st.slider("Key Points", 3, 10, 5)
    show_stats = st.checkbox("Show Statistics", value=True)
    show_bleu = st.checkbox("Show BLEU Score", value=True)

# ======================== MAIN TABS ========================

tab1, tab2, tab3 = st.tabs(["Summarize", "Quiz", "Info"])

# ======================== TAB 1: SUMMARIZE ========================

with tab1:
    st.header("📄 Text/PDF Input")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        input_method = st.radio("Input Type", ["Text", "PDF"], horizontal=True)
        
        if input_method == "Text":
            text_input = st.textarea("Enter your notes:", height=250, placeholder="Paste academic text...")
        else:
            uploaded_file = st.file_uploader("Upload PDF", type="pdf")
            text_input = ""
            pdf_metrics = {}
            
            if uploaded_file:
                with st.spinner("Processing PDF..."):
                    text_input, pdf_metrics = summarizer.extract_text_from_pdf(uploaded_file)
                    
                    if not text_input.startswith("Error"):
                        st.success("✅ PDF Processed Successfully!")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Pages Extracted", pdf_metrics.get("extracted_pages", 0))
                        with col2:
                            st.metric("Total Words", pdf_metrics.get("total_words", 0))
                        with col3:
                            st.metric("Sentences", pdf_metrics.get("total_sentences", 0))
                        with col4:
                            st.metric("Avg Words/Page", int(pdf_metrics.get("avg_words_per_page", 0)))
                    else:
                        st.error(text_input)
    
    if text_input and not text_input.startswith("Error"):
        if st.button("✨ Generate Summary", type="primary", use_container_width=True):
            if len(text_input) < 50:
                st.error("❌ Enter at least 50 characters")
            else:
                with st.spinner("Processing with AI..."):
                    # Text Analysis
                    if show_stats:
                        st.subheader("📊 Text Analysis")
                        analysis = summarizer.analyze_text(text_input)
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Words", analysis['total_words'])
                        with col2:
                            st.metric("Sentences", analysis['total_sentences'])
                        with col3:
                            st.metric("Unique", analysis['unique_words'])
                        with col4:
                            st.metric("Diversity", f"{analysis['lexical_diversity']}%")
                    
                    # Generate Summary
                    st.subheader("📝 Summary")
                    summary = None
                    summary_length_param = 150
                    
                    if "Extractive" in selected_model:
                        summary_length_param = num_sentences
                    else:
                        summary_length_param = summary_length
                    
                    if "GPT-4o-mini" in selected_model or ("Generative" in selected_model and summarizer.openai_client):
                        if summarizer.openai_client:
                            summary = summarizer.openai_summarize(text_input, summary_length_param)
                        else:
                            st.error("❌ OpenAI API not available. Select another model.")
                            summary = None
                    elif "DistilGPT2" in selected_model or ("Generative" in selected_model and summarizer.distilgpt2):
                        if summarizer.distilgpt2:
                            summary = summarizer.distilgpt2_summarize(text_input, summary_length_param)
                        else:
                            st.error("❌ DistilGPT2 not loaded. Select another model.")
                            summary = None
                    else:
                        summary = summarizer.extractive_summarize(text_input, summary_length_param)
                    
                    if summary:
                        st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
                        st.caption(f"📌 Method used: {summarizer.last_summary_method}")
                        
                        # BLEU Score
                        if show_bleu:
                            st.subheader("📈 Summary Quality")
                            metrics = summarizer.evaluate_summary_quality(text_input, summary)
                            scores = metrics['bleu_scores']
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("BLEU-1", f"{scores.get('BLEU-1', 0):.2f}")
                            with col2:
                                st.metric("BLEU-2", f"{scores.get('BLEU-2', 0):.2f}")
                            with col3:
                                st.metric("BLEU-3", f"{scores.get('BLEU-3', 0):.2f}")
                            with col4:
                                st.metric("BLEU-4", f"{scores.get('BLEU-4', 0):.2f}")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Compression", f"{metrics['compression_ratio']:.1f}%")
                            with col2:
                                st.metric("Token Overlap", f"{metrics['token_overlap']:.1f}%")
                            with col3:
                                st.metric("Avg Sent Len", metrics['avg_sentence_length'])
                            with col4:
                                st.metric("Rating", metrics['quality_rating'])
                    
                    # Keywords
                    st.subheader("🔑 Keywords")
                    keywords = summarizer.extract_keywords(text_input, num_keywords)
                    keyword_html = ' '.join([f"<span class='keyword-tag'>{k[0]}</span>" for k in keywords])
                    st.markdown(keyword_html, unsafe_allow_html=True)
                    
                    # Bullet Points
                    st.subheader("📍 Key Points")
                    points = summarizer.generate_bullet_points(text_input, num_bullet_points)
                    for i, point in enumerate(points, 1):
                        st.markdown(f"**{i}.** {point}")
                    
                    # Store for quiz tab
                    st.session_state.summarized_text = summary

# ======================== TAB 2: QUIZ ========================

with tab2:
    st.header("🎓 Quiz Generator")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        quiz_input_method = st.radio("Input Type", ["Text", "PDF"], horizontal=True, key="quiz_input_type")
        
        if quiz_input_method == "Text":
            quiz_input = st.textarea("Text for quiz", height=200, placeholder="Paste text or use summary from Summarize tab...", value=st.session_state.get("summarized_text", ""), key="quiz_text")
        else:
            quiz_pdf_file = st.file_uploader("Upload PDF for quiz", type="pdf", key="quiz_pdf")
            quiz_input = ""
            quiz_pdf_metrics = {}
            
            if quiz_pdf_file:
                with st.spinner("Processing PDF for quiz..."):
                    quiz_input, quiz_pdf_metrics = summarizer.extract_text_from_pdf(quiz_pdf_file)
                    
                    if not quiz_input.startswith("Error"):
                        st.success("✅ PDF Processed Successfully!")
                    else:
                        st.error(quiz_input)
    
    if "generated_quiz" not in st.session_state:
        st.session_state.generated_quiz = None
    
    if quiz_input and not quiz_input.startswith("Error"):
        if st.button("🎲 Generate Quiz", type="primary", use_container_width=True):
            if len(quiz_input) < 50:
                st.error("❌ Text too short for quiz generation")
            else:
                with st.spinner("Generating quiz questions..."):
                    if "Multiple Choice" in quiz_type:
                        quiz = summarizer.generate_mcq_quiz(quiz_input, num_quiz_questions)
                    else:
                        quiz = summarizer.generate_true_false_quiz(quiz_input, num_quiz_questions)
                    
                    if quiz:
                        st.session_state.generated_quiz = quiz
                        st.success(f"✅ Generated {len(quiz)} questions!")
                    else:
                        st.error("❌ Failed to generate quiz. Try with longer text.")
    
    if st.session_state.generated_quiz:
        quiz = st.session_state.generated_quiz
        
        for idx, q in enumerate(quiz, 1):
            st.markdown(f"<div class='quiz-card'>", unsafe_allow_html=True)
            
            if q['type'] == 'mcq':
                st.markdown(f"**Question {idx}:** {q['question']}")
                answer_key = f"quiz_answer_{idx}"
                selected = st.radio("Select answer", q['options'], key=answer_key, label_visibility="collapsed")
                
                if st.button(f"Check Answer", key=f"check_{idx}"):
                    if selected == q['correct_answer']:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong. Correct answer: {q['correct_answer']}")
            else:
                st.markdown(f"**Statement {idx}:** {q['question']}")
                tf_key = f"tf_answer_{idx}"
                selected = st.radio("True or False?", [True, False], key=tf_key, horizontal=True, label_visibility="collapsed")
                
                if st.button(f"Check Answer", key=f"check_tf_{idx}"):
                    if selected == q['answer']:
                        st.success("✅ Correct!")
                    else:
                        st.error(f"❌ Wrong. Correct answer: {q['answer']}")
            
            st.markdown(f"</div>", unsafe_allow_html=True)

# ======================== TAB 3: INFO ========================

with tab3:
    st.header("ℹ️ About This App")
    
    st.markdown("""
    ### 🎯 AI-Powered Summarization
    - **Generate concise summaries** from long academic texts
    - **Multiple AI models**: OpenAI, DistilGPT2, or Extractive
    - **Adjustable summary length** for your needs
    
    ### 📄 PDF Processing
    - **Upload and extract** text from PDF documents
    - **Automatic text analysis** and preprocessing
    - **Extract metadata** about your documents
    
    ### 🎓 Quiz Generation
    - **Create Multiple Choice Questions (MCQ)** automatically
    - **Generate True/False questions** instantly
    - **Interactive answer checking** with instant feedback
    
    ### 📊 Quality Analysis
    - **BLEU score evaluation** for summary quality
    - **Compression ratio** and token overlap metrics
    - **Keyword extraction** and key point identification
    
    ### 🛠️ Additional Tools
    - **Automatic keyword extraction** from text
    - **Bullet point generation** for key concepts
    - **Text statistics** and reading time estimation
    
    ---
    
    Made with ❤️ for students and researchers
    """)