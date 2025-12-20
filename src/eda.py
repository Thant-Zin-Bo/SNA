"""
eda.py
-------------------------
Exploratory Data Analysis module for US Election 2020 SNA Project.
Matches Master-level requirements for Data Mining.

Functions:
1. load_data: Robust CSV loading.
2. extract_interactions: Core Regex logic for network edges.
3. analyze_topology_feasibility (RQ1): Checks giant components & sparsity.
4. analyze_semantic_feasibility (RQ2): Checks text length & duplication for LDA.
5. analyze_multiplex_overlap (RQ3): Checks user overlap between RT & Mention layers.
6. plot_temporal_and_user_stats: General volume and power-law checks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import re
import sys

# Set visualization style
sns.set(style="whitegrid")

def load_data(filepath, limit=None):
    """
    Loads the dataset handling specific parsing issues (newlines in tweets).
    
    Args:
        filepath (str): Path to the CSV.
        limit (int, optional): Load only N rows for testing.
        
    Returns:
        pd.DataFrame: Loaded dataframe or None if failed.
    """
    print(f"📂 Loading data from: {filepath}...")
    try:
        # lineterminator='\n' is REQUIRED for this specific Kaggle dataset
        df = pd.read_csv(
            filepath, 
            lineterminator='\n', 
            parse_dates=['created_at'],
            nrows=limit,
            dtype={'user_id': str, 'tweet_id': str} # Prevent ID truncation
        )
        print(f"✅ Loaded {len(df):,} tweets.")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def extract_interactions(text):
    """
    Parses tweet text to find edges.
    Returns: (list_of_retweet_targets, list_of_mention_targets)
    """
    if not isinstance(text, str):
        return [], []
    
    # 1. Extract Retweets (Pattern: "RT @username")
    rt_pattern = r'RT @(\w+)'
    rts = re.findall(rt_pattern, text)
    
    # 2. Extract Mentions (Pattern: "@username")
    # We remove the RT part first to avoid counting the retweeted user as a mention
    text_no_rt = re.sub(rt_pattern, '', text)
    mention_pattern = r'@(\w+)'
    mentions = re.findall(mention_pattern, text_no_rt)
    
    return rts, mentions

# ==============================================================================
# RQ1: TOPOLOGY CHECKS
# ==============================================================================

def analyze_topology_feasibility(df, sample_size=10000):
    """
    [RQ1 Check] Estimates if the graph is connected enough for Modularity analysis.
    Checks for the existence of a 'Giant Component'.
    """
    print("\n📊 --- RQ1: Topology Feasibility Check ---")
    
    # Sample for speed
    if len(df) > sample_size:
        sample = df.sample(sample_size, random_state=42)
    else:
        sample = df

    G = nx.DiGraph()
    edge_count = 0
    
    print(f"   Building preview graph from {len(sample)} tweets...")
    
    for _, row in sample.iterrows():
        user = row['user_screen_name']
        rts, _ = extract_interactions(str(row['tweet']))
        
        for target in rts:
            G.add_edge(user, target)
            edge_count += 1
            
    if edge_count == 0:
        print("   ❌ No edges found in sample. Regex might be failing.")
        return

    # Check Connectivity (Weakly connected for directed graphs)
    components = list(nx.weakly_connected_components(G))
    if not components:
        print("   ❌ Graph empty.")
        return

    giant_size = len(max(components, key=len))
    total_nodes = len(G)
    
    print(f"   - Nodes in sample: {total_nodes}")
    print(f"   - Edges in sample: {edge_count}")
    print(f"   - Connected Components: {len(components)}")
    print(f"   - Giant Component Size: {giant_size} ({giant_size/total_nodes:.1%})")
    
    if (giant_size / total_nodes) < 0.05:
        print("   ⚠️ WARNING: Network is highly fragmented. 'Echo Chamber' detection may fail.")
    else:
        print("   ✅ Dominant Giant Component detected. Modularity analysis is valid.")

# ==============================================================================
# RQ2: SEMANTIC CHECKS
# ==============================================================================

def analyze_semantic_feasibility(df):
    """
    [RQ2 Check] Checks text richness for LDA Topic Modeling.
    Warns if data is too short or too repetitive (bot spam).
    """
    print("\nABC --- RQ2: Semantic Feasibility Check ---")
    
    # Calculate word counts
    df['word_count'] = df['tweet'].astype(str).apply(lambda x: len(x.split()))
    avg_len = df['word_count'].mean()
    
    # Check for duplicates (Bot/Spam check)
    unique_ratio = df['tweet'].nunique() / len(df)
    
    print(f"   - Avg Words per Tweet: {avg_len:.1f}")
    print(f"   - Unique Content Ratio: {unique_ratio:.1%}")
    
    # Heuristics
    if avg_len < 8:
        print("   ⚠️ WARNING: Tweets are very short. LDA models need rich text.")
    if unique_ratio < 0.3:
        print("   ⚠️ CRITICAL: Massive duplication detected. Aggressive de-duplication required.")
    else:
        print("   ✅ Text richness looks sufficient for Topic Modeling.")

# ==============================================================================
# RQ3: MULTIPLEXITY CHECKS
# ==============================================================================

def analyze_multiplex_overlap(df, sample_size=20000):
    """
    [RQ3 Check] Checks if the same users appear in both layers.
    If overlap is 0, you cannot compare behaviors for the same person.
    """
    print("\n⚔️  --- RQ3: Multiplex Feasibility Check ---")
    
    sample = df.sample(min(len(df), sample_size), random_state=42)
    
    # Sets to store user screen_names
    rt_layer_users = set()
    mention_layer_users = set()
    
    for _, row in sample.iterrows():
        user = row['user_screen_name']
        text = str(row['tweet'])
        
        # Check if this user PERFORMED an action
        rts, mentions = extract_interactions(text)
        
        if rts:
            rt_layer_users.add(user)
        if mentions:
            mention_layer_users.add(user)
            
    # Jaccard of USERS
    intersect = len(rt_layer_users.intersection(mention_layer_users))
    union = len(rt_layer_users.union(mention_layer_users))
    
    if union == 0:
        print("   ❌ No active users found.")
        return

    jaccard = intersect / union
    
    print(f"   - Users Retweeting: {len(rt_layer_users)}")
    print(f"   - Users Mentioning: {len(mention_layer_users)}")
    print(f"   - Users doing BOTH: {intersect}")
    print(f"   - User Overlap (Jaccard): {jaccard:.4f}")
    
    if intersect < 50:
        print("   ⚠️ WARNING: Very few users do both. Correlation analysis will be weak.")
    else:
        print("   ✅ Sufficient overlap to test 'Battlefield vs. Echo Chamber'.")

# ==============================================================================
# GENERAL STATS (TIME & BOTS)
# ==============================================================================

def plot_general_stats(df, label):
    """
    Plots temporal volume and user activity distribution.
    """
    print(f"\n📈 Generating General Stats for {label}...")
    
    # 1. Temporal Plot
    daily = df.set_index('created_at').resample('D').size()
    
    plt.figure(figsize=(12, 5))
    daily.plot(linewidth=2, color='navy')
    plt.title(f'Daily Tweet Volume: {label}')
    plt.ylabel('Tweets')
    plt.xlabel('Date')
    plt.tight_layout()
    plt.show()
    
    # 2. Power Law / Bot Check
    user_counts = df['user_id'].value_counts()
    
    plt.figure(figsize=(8, 5))
    plt.loglog(range(len(user_counts)), user_counts.values, marker='.', linestyle='none', alpha=0.3, color='crimson')
    plt.title(f'User Activity Distribution: {label}')
    plt.xlabel('User Rank (Log)')
    plt.ylabel('Tweets Posted (Log)')
    plt.tight_layout()
    plt.show()
    
    # Stats
    top_1 = user_counts.head(int(len(user_counts)*0.01)).sum()
    print(f"   - Top 1% Users generated {top_1/len(df):.1%} of all content.")