# DSS5104 Assignment 2 - 
TEXT CLASSIFICATION BENCHMARK

> **TEAM MEMBERS**: 
Cheng Yi Zhen (A0333501W)
Chua Wei Lin (A0170134U)
Rita Indah Lestari (A0115171U)
Repo**: [thatrita/DSS5104-Assignment-2](https://github.com/thatrita/DSS5104-Assignment-2)

> **PROJECT OVERVIEW**: 
This project implements a tiered text classification benchmark comparing:
- **Tier 1**: Classical ML (TF-IDF + Logistic Regression / SVM)
- **Tier 2**: Neural architecture (BiLSTM with custom training loop)
- **Tier 3**: Transformer fine-tuning (RoBERTa-base, DistilBERT-base)
- **Tier 4**: Data efficiency analysis using SetFit for few-shot learning

**Datasets**:
| Dataset | Task | Classes | Train/Val/Test |
|---------|------|---------|---------------|
| **20 Newsgroups** | Multi-class topic classification | 20 | 2,442 / 305 / 306 |
| **TweetEval-Irony** | Binary irony detection | 2 (ironic/non-ironic) | 2,862 / 955 / 784 |

**Primary Metric**: Macro F1-score (to handle class imbalance)  
**Robustness**: Multi-seed evaluation (seeds: 42, 123, 7) with mean ± std reporting

# 🗂️ Repository Structure
```text
DSS5104-Assignment-2/
├── notebooks/
│   ├── 01_Main Analysis and Tier 1.ipynb      # Classical models + BiLSTM
│   ├── 02_Tier 2 Roberta _ Tier 3 Distilbert Seed 7.ipynb
│   ├── 03_Tier 3 Distilbert_Seed 42 and 103 .ipynb
│   └── 04_Data Efficiency_setfit.ipynb        # Few-shot SetFit analysis
├── results/
│   ├── figures/          # Saved plots (cost-accuracy, F1 comparison, heatmaps)
│   ├── models/           # Serialized checkpoints (.pt)
│   └── metrics/          # CSV summaries of evaluation results
├── data/                 # NOT TRACKED – download via Kaggle API
├── requirements.txt      # Pinned dependencies for reproducibility
├── README.md             # This file
└── LICENSE               # MIT License

## 🚀 How to Reproduce

### Prerequisites
```bash
# Python 3.10+, GPU recommended for transformer experiments
# Kaggle API credentials for dataset download (~/.kaggle/kaggle.json)
```
###Installation
```bash
git clone https://github.com/thatrita/DSS5104-Assignment-2.git
cd DSS5104-Assignment-2
pip install -r requirements.txt
```
### Data Download (handled in notebook)
```bash
import kagglehub
# 20 Newsgroups
kagglehub.dataset_download("crawford/20-newsgroups")
# TweetEval Irony
kagglehub.dataset_download("thedevastator/tweeteval-a-multi-task-classification-benchmark")
```

### Run Notebooks
```bash
jupyter lab notebooks/01_Main\ Analysis\ and\ Tier\ 1.ipynb
