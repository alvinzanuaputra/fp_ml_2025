# CONTOH: Cara Load dan Menggunakan Model yang Sudah Disimpan

import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import re
import string

# ============================================================================
# 1. LOAD SEMUA MODEL DAN VECTORIZER
# ============================================================================

print("Loading saved models...")

# Load TF-IDF Vectorizer
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
print("✓ TF-IDF Vectorizer loaded")

# Load SVM Model
svm_model = joblib.load('svm_sentiment_model.pkl')
print("✓ SVM Model loaded")

# Load XGBoost Model
xgb_model = joblib.load('xgb_sentiment_model.pkl')
print("✓ XGBoost Model loaded")

# Load ANN Model
ann_model = load_model('ann_sentiment_model.h5')
print("✓ ANN Model loaded")

# ============================================================================
# 2. FUNGSI PREPROCESSING (HARUS SAMA DENGAN SAAT TRAINING)
# ============================================================================

def preprocess_text(text):
    """
    Fungsi preprocessing yang SAMA dengan yang digunakan saat training
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# ============================================================================
# 3. FUNGSI PREDIKSI SENTIMEN
# ============================================================================

def predict_sentiment(text):
    """
    Fungsi untuk memprediksi sentimen dari teks baru
    """
    # Preprocessing
    processed_text = preprocess_text(text)
    
    # Vectorization menggunakan TF-IDF yang sudah di-fit
    text_vectorized = tfidf_vectorizer.transform([processed_text])
    text_dense = text_vectorized.toarray()
    
    # Prediksi dengan semua model
    svm_pred = svm_model.predict(text_vectorized)[0]
    svm_confidence = abs(svm_model.decision_function(text_vectorized)[0])
    
    xgb_pred = xgb_model.predict(text_vectorized)[0]
    xgb_confidence = xgb_model.predict_proba(text_vectorized)[0].max()
    
    ann_pred = (ann_model.predict(text_dense) > 0.5).astype(int)[0][0]
    ann_confidence = ann_model.predict(text_dense)[0][0]
    ann_confidence = ann_confidence if ann_pred == 1 else (1 - ann_confidence)
    
    # Convert to readable format
    def convert_prediction(pred):
        return 'Positive' if pred == 1 else 'Negative'
    
    return {
        'text': text,
        'svm': {
            'prediction': convert_prediction(svm_pred),
            'confidence': svm_confidence
        },
        'xgboost': {
            'prediction': convert_prediction(xgb_pred),
            'confidence': xgb_confidence
        },
        'ann': {
            'prediction': convert_prediction(ann_pred),
            'confidence': ann_confidence
        }
    }

# ============================================================================
# 4. FUNGSI PREDIKSI BATCH (BANYAK TEKS SEKALIGUS)
# ============================================================================

def predict_sentiment_batch(texts):
    """
    Fungsi untuk memprediksi sentimen banyak teks sekaligus
    """
    results = []
    for text in texts:
        result = predict_sentiment(text)
        results.append(result)
    return results

# ============================================================================
# 5. CONTOH PENGGUNAAN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("          SENTIMENT ANALYSIS - USING SAVED MODELS")
    print("="*60)
    
    # Test dengan teks baru
    new_texts = [
        "This movie is absolutely fantastic! I loved every moment.",
        "Terrible film, complete waste of time and money.",
        "It was okay, nothing special but not bad either.",
        "The acting was brilliant but the story was confusing.",
        "I didn't like it at all, very disappointing."
    ]
    
    print("\n🔍 ANALYZING NEW TEXTS:")
    
    for i, text in enumerate(new_texts, 1):
        print(f"\n--- Text {i} ---")
        result = predict_sentiment(text)
        
        print(f"Text: '{text}'")
        print(f"SVM:      {result['svm']['prediction']:8s} (confidence: {result['svm']['confidence']:.3f})")
        print(f"XGBoost:  {result['xgboost']['prediction']:8s} (confidence: {result['xgboost']['confidence']:.3f})")
        print(f"ANN:      {result['ann']['prediction']:8s} (confidence: {result['ann']['confidence']:.3f})")
    
    # ========================================================================
    # 6. INTERACTIVE MODE - USER INPUT
    # ========================================================================
    
    print(f"\n{'='*60}")
    print("🎯 INTERACTIVE SENTIMENT ANALYSIS")
    print("Enter your text to analyze sentiment (or 'quit' to exit)")
    print("="*60)
    
    while True:
        user_input = input("\nEnter text: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text!")
            continue
        
        try:
            result = predict_sentiment(user_input)
            
            print(f"\n📊 SENTIMENT ANALYSIS RESULTS:")
            print(f"Text: '{user_input}'")
            print(f"┌─────────────────────────────────────┐")
            print(f"│ SVM:      {result['svm']['prediction']:8s} ({result['svm']['confidence']:.3f}) │")
            print(f"│ XGBoost:  {result['xgboost']['prediction']:8s} ({result['xgboost']['confidence']:.3f}) │")
            print(f"│ ANN:      {result['ann']['prediction']:8s} ({result['ann']['confidence']:.3f}) │")
            print(f"└─────────────────────────────────────┘")
            
        except Exception as e:
            print(f"❌ Error: {e}")

# ============================================================================
# 7. FUNGSI UNTUK WEB APPLICATION (FLASK/DJANGO)
# ============================================================================

class SentimentAnalyzer:
    """
    Class untuk digunakan dalam web application
    """
    def __init__(self):
        self.tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
        self.svm_model = joblib.load('svm_sentiment_model.pkl')
        self.xgb_model = joblib.load('xgb_sentiment_model.pkl')
        self.ann_model = load_model('ann_sentiment_model.h5')
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join(text.split())
        return text
    
    def predict(self, text, model_type='svm'):
        """
        Prediksi dengan model tertentu
        model_type: 'svm', 'xgboost', or 'ann'
        """
        processed_text = self.preprocess_text(text)
        text_vectorized = self.tfidf_vectorizer.transform([processed_text])
        
        if model_type == 'svm':
            prediction = self.svm_model.predict(text_vectorized)[0]
            confidence = abs(self.svm_model.decision_function(text_vectorized)[0])
        elif model_type == 'xgboost':
            prediction = self.xgb_model.predict(text_vectorized)[0]
            confidence = self.xgb_model.predict_proba(text_vectorized)[0].max()
        elif model_type == 'ann':
            text_dense = text_vectorized.toarray()
            prediction = (self.ann_model.predict(text_dense) > 0.5).astype(int)[0][0]
            prob = self.ann_model.predict(text_dense)[0][0]
            confidence = prob if prediction == 1 else (1 - prob)
        
        sentiment = 'Positive' if prediction == 1 else 'Negative'
        return {'sentiment': sentiment, 'confidence': float(confidence)}

# Contoh penggunaan untuk web app:
# analyzer = SentimentAnalyzer()
# result = analyzer.predict("This is a great movie!", model_type='svm')
# print(result)  # {'sentiment': 'Positive', 'confidence': 0.85}

print("\n✅ All functions ready for use!")
print("💡 You can now use these models in your applications!")