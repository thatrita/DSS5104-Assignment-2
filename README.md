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
│   ├── 01_Main Analysis and Tier 1 & 2.ipynb           
│   ├── 02_Tier 2 Roberta _ Tier 3 Distilbert Seed 7.ipynb
│   ├── 03_Tier 3 Distilbert_Seed 42 and 103 .ipynb
│   └── 04_Data Efficiency_setfit.ipynb     
├── results/
│   ├── all_results_distil_seed7.pkl                    # DistilBERT results (seed 7)
│   ├── all_results_distil_seed42.pkl                   # DistilBERT results (seed 42)
│   ├── all_results_distil_seed123.pkl                  # DistilBERT results (seed 123)
│   ├── all_results_roberta.pkl                         # RoBERTa model results
│   ├── BiLSTM (Tier 2 Neural Model) and Classical Models (TF-IDR + LR / SVM) result.pkl
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

## 📊 Results Summary

### 20 Newsgroups (Multi-class Classification)
| Model | Tier | Accuracy (±std) | Macro F1 (±std) | Train Time |
|-------|------|-----------------|-----------------|------------|
| TF-IDF + LR | 1 | 0.902 ± 0.000 | 0.889 ± 0.000 | 1.0s |
| TF-IDF + SVM | 1 | 0.905 ± 0.000 | 0.893 ± 0.000 | 1.3s |
| BiLSTM | 2 | **0.820 ± 0.023** | **0.816 ± 0.023** | 188.4s |
| DistilBERT | 3 | **0.877 ± 0.012** | **0.863 ± 0.010** | 2432.7s |
| RoBERTa-base | 3 | **0.851 ± 0.004** | **0.820** | 980.4s ± 139.7s |

### TweetEval-Irony (Binary Classification)
| Model | Tier | Accuracy (±std) | Macro F1 (±std) | Train Time |
|-------|------|-----------------|-----------------|------------|
| TF-IDF + LR | 1 | 0.653 ± 0.000 | 0.643 ± 0.000 | 0.05s |
| TF-IDF + SVM | 1 | 0.652 ± 0.000 | 0.642 ± 0.000 | 0.06s |
| BiLSTM | 2 | **0.608 ± 0.009** | **0.584 ± 0.022** | 51.1s |
| DistilBERT | 3 | **0.682 ± 0.015** | **0.674 ± 0.017** | 1509.8s |
| RoBERTa-base | 3 | **0.704 ± 0.013** | **0.701** | 723.9s ± 21.1s |

🔍 **Key Insight**: Classical models achieve strong performance on 20NG with minimal compute. BiLSTM shows moderate gains but requires ~188s training. DistilBERT balances accuracy and efficiency, while RoBERTa excels on the semantically subtle irony task (F1=0.701). The `talk.religion.misc` and `ironic` classes remain challenging across all architectures.
```

## 🚀 How to Reproduce

### Prerequisites
Python 3.10+, GPU recommended for transformer experiments
Kaggle API credentials for dataset download (~/.kaggle/kaggle.json)

### Installation
git clone https://github.com/thatrita/DSS5104-Assignment-2.git
cd DSS5104-Assignment-2
pip install -r requirements.txt


### Data Download (handled in notebook)
import kagglehub
# 20 Newsgroups
kagglehub.dataset_download("crawford/20-newsgroups")
# TweetEval Irony
kagglehub.dataset_download("thedevastator/tweeteval-a-multi-task-classification-benchmark")

### Run Notebooks
jupyter lab notebooks/01_Main\ Analysis\ and\ Tier\ 1.ipynb
```