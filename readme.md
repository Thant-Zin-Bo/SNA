# Multiplex Dynamics of Polarization: US Election 2020 SNA

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![NetworkX](https://img.shields.io/badge/NetworkX-3.1-green)
![Status](https://img.shields.io/badge/Status-Phase%202%3A%20Graph%20Construction-orange)

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
│   ├── raw/             # (Ignored) Original Kaggle CSVs
│   ├── processed/       # Intermediate cleaned data
│   └── graphs/          # .gexf files for Gephi visualization
├── notebooks/
│   ├── 01_eda.ipynb                # Feasibility checks & Power Law distribution
│   └── 02_topology_comparison.ipynb # Graph construction & Metric calculation
├── src/
│   ├── eda.py              # Data loading & Feasibility logic
│   ├── network_analysis.py # Graph building & Topology metrics
│   └── text_analysis.py    # (Planned) LDA Topic Modeling
└── requirements.txt     # Python dependencies
🚀 Getting Started
1. Clone the Repository
Bash

git clone [https://github.com/Thant-Zin-Bo/SNA.git](https://github.com/Thant-Zin-Bo/SNA.git)
cd SNA
2. Set Up Environment
It is recommended to use a virtual environment to manage dependencies.

Mac / Linux:

Bash

python3 -m venv .venv
source .venv/bin/activate
Windows:

Bash

python -m venv .venv
.venv\Scripts\activate
Install Requirements:

Bash

pip install -r requirements.txt
python -m spacy download en_core_web_sm
3. Download Data
⚠️ Note: The dataset is not included in this repo due to size constraints (GitHub limit 100MB).

Download the US Election 2020 Tweets dataset from Kaggle.

Create a folder named data/raw/.

Place the following files inside:

hashtag_donaldtrump.csv

hashtag_joebiden.csv

4. Run Analysis
Step 1: Open notebooks/01_eda.ipynb to verify data quality and check for "Giant Components."

Step 2: Open notebooks/02_topology_comparison.ipynb to run the Modularity calculation and export graphs for Gephi.

📊 Methodology & Tools
Edge Extraction: Custom Regex parsing (in src/eda.py) to separate RT @User (Endorsement) from @User (Discussion).

Graph Theory: NetworkX for calculating Modularity (Louvain method) and Density.

NLP: Gensim for LDA Topic Modeling.

Visualization: Gephi (ForceAtlas2 layout) will be used for final network maps.

👥 Contributors
Thant Zin Bo - Data Engineering & Topology Analysis

[Abhinav Ramalingam] - 

[Moutushi Sen] - 
