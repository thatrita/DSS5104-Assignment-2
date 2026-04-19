# DSS5104 Assignment 2: Text Classification & Model Comparison

> **Course:** DSS5104 — Machine Learning and Predictive Modelling  
> **Assignment:** 2 — Text Classification  
> **Focus:** Reproducible comparison of classical ML, neural, and transformer models

> **TEAM MEMBERS**:  
Cheng Yi Zhen (A0333501W)  
Chua Wei Lin (A0170134U)  
Rita Indah Lestari (A0115171U)  
Repo**: [thatrita/DSS5104-Assignment-2](https://github.com/thatrita/DSS5104-Assignment-2)

---

## 🎯 Project Overview

This project implements and evaluates multiple text classification approaches:

| Tier | Model | Description |
|------|-------|-------------|
| **Tier 1** | TF-IDF + Logistic Regression / SVM | Classical baselines with sparse features |
| **Tier 2** | BiLSTM | Recurrent neural network with embeddings |
| **Tier 2** | RoBERTa-base | Pre-trained transformer (fine-tuned) |
| **Tier 3** | DistilBERT-base-uncased | Lightweight transformer (fine-tuned, multiple seeds) |
| **Bonus** | SetFit | Few-shot learning framework |

**Datasets:** 20 Newsgroups, TweetEval  
**Metrics:** F1-score (macro/weighted), Accuracy, Inference Time, Training Time, Cost-Accuracy Tradeoff
**Robustness**: Multi-seed evaluation (seeds: 42, 123, 7) with mean ± std reporting

# 🗂️ Repository Structure
```text
DSS5104-Assignment-2/
├── notebooks/
│   ├── 01_Main Analysis and Tier 1 & 2.ipynb           
│   ├── 02_Tier 2 Roberta _ Tier 3 Distilbert Seed 7.ipynb
│   ├── 03_Tier 3 Distilbert_Seed 42 and 103 .ipynb
│   └── 04_Data Efficiency_setfit.ipynb     
├── results/
│   ├── all_results_distil_seed7.pkl                    # DistilBERT results (seed 7)
│   ├── all_results_distil_seed42.pkl                   # DistilBERT results (seed 42)
│   ├── all_results_distil_seed123.pkl                  # DistilBERT results (seed 123)
│   ├── all_results_roberta.pkl                         # RoBERTa model results
│   ├── dss5104asg2_wl12042026_outputs.pkl              # TF-IDF + LR _ SVM and BiLSTM results
│   ├── figures/
│   │   ├── results/
│   │   │   ├── cost_accuracy_tradeoff.png              # Cost vs accuracy analysis
│   │   │   ├── inference_time.png                      # Inference time comparison
│   │   │   ├── model_comparison_f1.png.png             # F1 score model comparison
│   │   │   ├── per_class_f1_heatmap_20ng.png           # Per-class F1 heatmap (20ng)
│   │   │   └── training_time_comparison_fixed.png      # Training time comparison
│   │   │   └── tweeteval_crossover_analysis.png        # TweetEval crossover analysis
│   │   │   └── tweeteval_crossover_plot.png            # TweetEval crossover plot
│   ├── tmp_final_distilbert-base-uncased_7/            # Final model checkpoint (seed 7)
│   ├── tmp_final_distilbert-base-uncased_42/           # Final model checkpoint (seed 42)
│   ├── tmp_final_distilbert-base-uncased_123/          # Final model checkpoint (seed 123)
│   ├── tmp_tune_distilbert-base-uncased_7/             # Tuning artifacts (seed 7)
│   ├── tmp_tune_distilbert-base-uncased_42/            # Tuning artifacts (seed 42)
│   └── tmp_tune_distilbert-base-uncased_123/           # Tuning artifacts (seed 123)   
├── data/                 # NOT TRACKED – download via Kaggle API
├── requirements.txt      # Pinned dependencies for reproducibility
├── README.md             # This file
└── LICENSE               # MIT License
```

To load results programmatically:
```python
import pickle
with open('results/all_results_distil_seed7.pkl', 'rb') as f:
    results = pickle.load(f)
```

---
## 📊 Results Summary

### 20 Newsgroups (Multi-class Classification)
| Model | Tier | Accuracy (±std) | Macro F1 (±std) | Train Time |
|-------|------|-----------------|-----------------|------------|
| TF-IDF + LR | 1 | 0.902 ± 0.000 | 0.889 ± 0.000 | 0.98 ± 0.02s |
| TF-IDF + SVM | 1 | 0.905 ± 0.000 | 0.893 ± 0.000 | 1.27 ± 0.00s |
| BiLSTM | 2 | 0.820 ± 0.023 | 0.816 ± 0.023 | 179.92 ± 4.03s |
| DistilBERT | 3 | 0.877 ± 0.012 | 0.863 ± 0.010 | 2432.67 ± 1678.68s |
| RoBERTa-base | 3 | 0.851 ± 0.004 | 0.830 ± 0.009 | 980.4s ± 139.74s |

### TweetEval-Irony (Binary Classification)
| Model | Tier | Accuracy (±std) | Macro F1 (±std) | Train Time |
|-------|------|-----------------|-----------------|------------|
| TF-IDF + LR | 1 | 0.653 ± 0.000 | 0.643 ± 0.000 | 0.05 ± 0.000s |
| TF-IDF + SVM | 1 | 0.652 ± 0.000 | 0.642 ± 0.000 | 0.06 ± 0.000s |
| BiLSTM | 2 | 0.608 ± 0.009 | 0.584 ± 0.022 | 54.91 ± 2.03s |
| DistilBERT | 3 | 0.682 ± 0.015 | 0.674 ± 0.017 | 1509.76 ± 970.70s |
| RoBERTa-base | 3 | 0.704 ± 0.013 | 0.689 ± 0.018 | 723.9s ± 21.1s |

🔍 **Key Insight**: Classical models achieve strong performance on 20NG with minimal compute. BiLSTM shows moderate gains but requires ~188s training. DistilBERT balances accuracy and efficiency, while RoBERTa excels on the semantically subtle irony task (F1=0.701). The `talk.religion.misc` and `ironic` classes remain challenging across all architectures.
```

```
## 🚀 How to Reproduce

### 1️⃣ Environment Setup


# Clone the repository
```bash
git clone https://github.com/thatrita/DSS5104-Assignment-2.git
cd DSS5104-Assignment-2
```

# Create virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

# Install dependencies
```bash
pip install -r requirements.txt
```


### 2️⃣ Data Preparation

Raw data is **not tracked in Git**. Download via:

```bash
# Option A: Kaggle API (for TweetEval/20ng if hosted)
kaggle datasets download -d <dataset-name> -p data/raw/
```
```bash
# Option B: HuggingFace Datasets (recommended for transformers)
# Handled automatically in notebooks via \`datasets.load_dataset()\`
```
Ensure data is placed in:
```bash
data/
├── raw/          # Original downloads
├── processed/    # Preprocessed CSV/JSON (auto-generated by notebooks)
└── README.md     # Data source documentation
```

