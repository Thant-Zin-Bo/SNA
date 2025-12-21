# Multiplex Dynamics of Polarization: US Election 2020 SNA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NetworkX](https://img.shields.io/badge/NetworkX-3.1-green)
![FastText](https://img.shields.io/badge/FastText-Language%20Filter-yellow)
![Status](https://img.shields.io/badge/Status-Phase%202%3A%20Data%20Preprocessing%20%26%20NLP-orange)

## 📌 Project Overview
This project mathematically quantifies the structure of political polarization on Twitter during the 2020 US Presidential Election. Moving beyond simple descriptive statistics, we employ a **Multiplex Network Approach** to test specific social theories about how "Echo Chambers" and "Battlefields" emerge in digital discourse.

We contrast the **Trump Support Network** (Treatment) against the **Biden Support Network** (Control) to isolate unique topological and semantic characteristics.

## 🔬 Research Questions
### 1. Topology (The "Bunker" Hypothesis)
* **Question:** To what extent does the Trump-related network exhibit higher structural modularity and density compared to the Biden-related network?
* **Metric:** Modularity ($Q$), Graph Density ($\rho$), and Average Clustering Coefficient ($C$).

### 2. Semantics (The "Framing" Hypothesis)
* **Question:** How effectively can semantic divergence be quantified using Topic Modeling, and does the linguistic gap correlate with structural clusters?
* **Metric:** Latent Dirichlet Allocation (LDA) & Jensen-Shannon Divergence.

### 3. Multiplexity (The "Battlefield" Hypothesis)
* **Question:** Do the "Endorsement Layer" (Retweets) and "Discussion Layer" (Mentions) exhibit low edge overlap, confirming that political networks function as dual-purpose structures?
* **Metric:** Jaccard Similarity Coefficient ($J$) between layer edge sets.

## 📂 Repository Structure

```text
├── data/
│   ├── raw/                       # Raw input CSVs (Local only)
│   ├── processed/                 # Cleaned Data Output
│   │   ├── *_lda_ready.csv        # Heavy Clean (No names/stopwords) for Topic Modeling
│   │   ├── *_bert_ready.csv       # Light Clean (Full sentences) for Sentiment/BERT
│   │   ├── *_bots_removed.csv     # Audit trail of removed bot accounts
│   │   └── *_foreign_removed.csv  # Audit trail of non-English tweets
│   └── graphs/                    # .gexf files for Gephi visualization
├── notebooks/
│   ├── 01_eda.ipynb               # Feasibility checks & Power Law distribution
│   ├── 02_preprocessing.ipynb     # MASTER PIPELINE: FastText -> Bot Filter -> NLP Fork
│   └── 03_topology.ipynb          # (Upcoming) Graph construction & Metrics
├── src/
│   ├── eda.py                     # Data loading & Feasibility logic
│   ├── preprocessing.py           # The Cleaning Engine (FastText, SpaCy, Bot Forensics)
│   └── network_analysis.py        # Graph building & Topology metrics
└── requirements.txt               # Python dependencies
🚀 Methodology: The "Forked Pipeline"To ensure scientific rigor, we process data through a strict 4-Layer Filter before analysis.Layer 1: Global Language Filter (FastText)Technology: Facebook's FastText (lid.176.ftz).Action: Removes non-English tweets (Portuguese, Turkish, German) that confuse Topic Models.Performance: ~100x faster than langdetect.Layer 2: Noise & Bot FiltrationNoise: Removes duplicates and short text (< 4 words).Bot Forensics: Identifies statistical outliers (Top 0.5% by volume) who exhibit non-human behavior (>144 tweets/day). These are removed to prevent skewing network centrality (RQ1).Layer 3: Semantic "Kill Switch"Action: Recursively removes candidate names (donaldtrump, sleepyjoebiden, kamala) from the text.Why: For RQ2, we measure how they talk (framing), not who they talk about.Layer 4: The Output ForkPath A (LDA Ready): Heavy cleaning. Lemmatization (voting $\to$ vote), Stopword removal.Path B (BERT Ready): Light cleaning. Preserves sentence structure and punctuation for deep learning context.🛠 Getting Started1. Clone the RepositoryBashgit clone [https://github.com/Thant-Zin-Bo/SNA.git](https://github.com/Thant-Zin-Bo/SNA.git)
cd SNA
2. Set Up EnvironmentIt is recommended to use a virtual environment.Mac / Linux:Bashpython3 -m venv .venv
source .venv/bin/activate
Windows:Bashpython -m venv .venv
.venv\Scripts\activate
Install Requirements:Bashpip install -r requirements.txt
python -m spacy download en_core_web_sm
3. Run AnalysisEnsure your raw data files (hashtag_donaldtrump.csv, hashtag_joebiden.csv) are located in data/raw/.
Step 1: PreprocessingOpen notebooks/02_preprocessing.ipynb. This notebook runs the full pipeline:Detects and removes foreign languages.Audits and removes Bots.Splits data into lda_ready and bert_ready formats.
Step 2: Topology (Next Phase)Open notebooks/03_topology.ipynb (Coming soon) to run Modularity calculations.

👥 ContributorsThant Zin Bo - Data Engineering & Topology Analysis
[Abhinav Ramalingam] -
[Moutushi Sen] -
