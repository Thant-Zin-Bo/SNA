"""
src/preprocessing.py
--------------------
Master-level preprocessing pipeline for SNA, LDA & BERT.
Target Architecture: CPU-Optimized (High Performance)
"""

import pandas as pd
import spacy
import sys
import os
import urllib.request
import re

# Check for FastText
try:
    import fasttext
except ImportError:
    print("❌ ERROR: 'fasttext' library is missing.")
    print("   Please run: pip install fasttext-wheel")
    sys.exit(1)

# --- GLOBAL MODEL LOADING ---
nlp = None
lang_model = None
LANG_MODEL_PATH = "lid.176.ftz"

try:
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    print("✅ spaCy model 'en_core_web_sm' loaded.")
except OSError:
    print("⚠️ WARNING: spaCy model not found! Run: python -m spacy download en_core_web_sm")

# --- CUSTOM STOPWORDS (For LDA) ---
CUSTOM_STOPWORDS = {
    'maga', 'potus', 'vp', 'campaign', 'candidate', 'poll',
    'vote', 'voting', 'election', '2020',
    'amp', 'https', 'http', 'rt', 'com', 'org', 'www', 'html',
    'people', 'time', 'day', 'country', 'america', 'american', 'nation',
    'state', 'year', 'man', 'woman', 'world', 'thing', 'way', 'news'
}

# --- FUNCTIONS ---

def download_fasttext_model():
    if not os.path.exists(LANG_MODEL_PATH):
        print(f"⬇️ Downloading FastText model...")
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"
        urllib.request.urlretrieve(url, LANG_MODEL_PATH)
        print("✅ Download complete.")

def filter_language(df, save_prefix=None):
    """Layer 0: Remove non-English tweets using FastText."""
    global lang_model
    if df is None or len(df) == 0: return df
    
    if lang_model is None:
        download_fasttext_model()
        fasttext.FastText.eprint = lambda x: None
        lang_model = fasttext.load_model(LANG_MODEL_PATH)
        print("✅ FastText model loaded.")

    print(f"   🌍 [Language Filter] Checking {len(df):,} tweets...")
    
    def is_english(text):
        clean_text = str(text).replace("\n", " ")
        try:
            predictions = lang_model.predict(clean_text)
            return predictions[0][0].replace("__label__", "") == 'en'
        except:
            return False

    mask = df['tweet'].apply(is_english)
    
    # Save Foreign Tweets
    if save_prefix:
        df_foreign = df[~mask].copy()
        if len(df_foreign) > 0:
            os.makedirs('../data/processed', exist_ok=True)
            save_path = f"../data/processed/{save_prefix}_foreign_removed.csv"
            df_foreign.to_csv(save_path, index=False)
            print(f"      -> 💾 Saved {len(df_foreign):,} foreign tweets to: {save_path}")
            
    df_eng = df[mask].copy()
    print(f"      -> Retained {len(df_eng):,} English tweets.")
    return df_eng

def filter_noise(df, min_words=4):
    """Layer 1: Remove duplicates and tiny tweets."""
    print(f"   🧹 [Noise Filter] Starting with {len(df):,} tweets...")
    df_clean = df.drop_duplicates(subset=['tweet'], keep='first').copy()
    df_clean['word_count'] = df_clean['tweet'].astype(str).apply(lambda x: len(x.split()))
    df_clean = df_clean[df_clean['word_count'] >= min_words]
    print(f"      -> Retained {len(df_clean):,} high-quality tweets.")
    return df_clean

def remove_bots(df, percentile=0.995, save_prefix=None):
    """
    Layer 2: Remove hyper-active users and SAVE them to CSV.
    """
    print(f"   🤖 [Bot Filter] Identifying top {(1-percentile)*100:.1f}% active users...")
    
    user_counts = df['user_id'].value_counts()
    threshold = user_counts.quantile(percentile)
    
    # Identify Bots
    bot_ids = user_counts[user_counts > threshold].index
    
    # Split Data
    df_bots = df[df['user_id'].isin(bot_ids)].copy()
    df_human = df[~df['user_id'].isin(bot_ids)].copy()
    
    print(f"      -> Found {len(bot_ids):,} bot accounts (Threshold > {int(threshold)} tweets).")
    print(f"      -> Removed {len(df_bots):,} tweets total.")
    
    # Save Removed Bots
    if save_prefix and len(df_bots) > 0:
        os.makedirs('../data/processed', exist_ok=True)
        save_path = f"../data/processed/{save_prefix}_bots_removed.csv"
        df_bots.to_csv(save_path, index=False)
        print(f"      -> 💾 Saved bot tweets to: {save_path}")
        
    return df_human

def analyze_bot_file(filepath):
    """
    Forensics Report: Loads a 'bots_removed.csv' file and verifies behavior.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found ({filepath})")
        return

    print(f"\n🕵️‍♂️ BOT FILE AUDIT: {os.path.basename(filepath)}")
    df_bots = pd.read_csv(filepath)
    print(f"   • Total Suspicious Tweets: {len(df_bots):,}")
    print(f"   • Total Suspicious Accounts: {df_bots['user_id'].nunique():,}")
    
    if len(df_bots) == 0: return

    # 1. Check the #1 Most Active Bot in this file
    user_counts = df_bots['user_id'].value_counts()
    top_bot_id = user_counts.index[0]
    top_bot_count = user_counts.iloc[0]
    
    # Speed Check
    user_tweets = df_bots[df_bots['user_id'] == top_bot_id]
    min_time = pd.to_datetime(user_tweets['created_at']).min()
    max_time = pd.to_datetime(user_tweets['created_at']).max()
    duration = (max_time - min_time).total_seconds() / 86400 # Days
    if duration < 0.1: duration = 0.1 # Prevent div by zero
    
    speed = top_bot_count / duration
    
    print(f"\n   🚩 TOP OFFENDER (User {top_bot_id}):")
    print(f"      - Posted {top_bot_count} times in {duration:.1f} days.")
    print(f"      - Speed: {speed:.1f} tweets/day")
    if speed > 144:
        print("      - 🤖 VERDICT: High Probability of Automation (10+ tweets/hour).")
        
    # 2. Source Check
    if 'source' in df_bots.columns:
        print(f"\n   🤖 Top Sources in this file:")
        print(df_bots['source'].value_counts().head(5).to_string())

    # 3. Content Check
    print(f"\n   📝 Sample Content:")
    print(user_tweets['tweet'].head(3).values)

def spacy_clean(texts):
    """Layer 4A: LDA Cleaning (Heavy)."""
    if nlp is None: raise RuntimeError("spaCy model not loaded.")
    print(f"   🧠 [LDA Prep] Heavy cleaning {len(texts):,} tweets...")
    docs = list(nlp.pipe(texts, batch_size=2000, n_process=1))
    cleaned_texts = []
    for doc in docs:
        tokens = []
        for token in doc:
            lemma = token.lemma_.lower()
            if 'trump' in lemma or 'biden' in lemma or 'harris' in lemma or 'pence' in lemma: continue
            if lemma in CUSTOM_STOPWORDS: continue
            if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and not token.is_stop:
                if token.is_alpha and len(lemma) > 2: tokens.append(lemma)
        cleaned_texts.append(" ".join(tokens))
    return cleaned_texts

def bert_clean(texts):
    """Layer 4B: BERT Cleaning (Light)."""
    print(f"   🤖 [BERT Prep] Light cleaning {len(texts):,} tweets...")
    cleaned_texts = []
    for text in texts:
        text = str(text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'\bRT\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        cleaned_texts.append(text)
    return cleaned_texts