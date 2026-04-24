# 🧪 ADME Property Predictor

**Predicted drug lipophilicity (logD) from molecular structure using classical ML and a fine-tuned ChemBERTa transformer, deployed as an interactive web app.**

Given a drug molecule as a [SMILES string](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), this project predicts lipophilicity (logD) — a key pharmacokinetic property that determines how a drug is absorbed and distributed in the body. XGBoost with combined molecular features achieved the best performance (R² = 0.704), outperforming a fine-tuned ChemBERTa transformer (R² = 0.531) on this small dataset.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

**[Try the Live Demo →](https://Nicholas-Adrogue-adme-predictor.hf.space)**

---

## Why This Matters

ADME properties determine whether a drug candidate will actually work in the human body. Poor pharmacokinetics is one of the leading causes of drug failure in clinical trials. Predicting these properties computationally saves time and money in early-stage drug discovery.

## Project Overview

This project takes an incremental approach — building from classical ML baselines to modern transformer-based models:

| Phase | Approach | Features | Status |
|-------|----------|----------|--------|
| 1 | Random Forest baseline | Morgan fingerprints | ✅ Complete |
| 2 | Feature engineering & model comparison | Morgan, MACCS, descriptors, combined | ✅ Complete |
| 3 | ChemBERTa fine-tuning | Learned SMILES representations | ✅ Complete |
| 4 | Demo & deployment | Interactive web app on HuggingFace Spaces | ✅ Complete |

## Results

| Phase | Model | Features | Test RMSE | Test R² |
|-------|-------|----------|-----------|---------|
| 1 | Random Forest | Morgan FP | 0.9255 | 0.3920 |
| 2 | Random Forest | Combined | 0.8077 | 0.5369 |
| 2 | **XGBoost** | **Combined** | **0.6457** | **0.7040** |
| 3 | ChemBERTa | SMILES (learned) | 0.8132 | 0.5306 |

XGBoost with combined features (Morgan fingerprints + physicochemical descriptors) achieved the best overall performance. ChemBERTa, despite being a much larger model, underperformed the best classical baseline — a result consistent with the literature on small molecular datasets (~4,000 compounds). The transformer's advantage typically emerges on datasets with 50,000+ molecules, where learned representations can surpass hand-crafted features.

## Key Findings

**Representation matters more than model complexity.** Moving from Morgan fingerprints alone (R² = 0.39) to combined features with descriptors (R² = 0.70) was a larger improvement than switching from Random Forest to XGBoost or ChemBERTa.

**Classical ML outperforms transformers on small datasets.** ChemBERTa showed clear signs of overfitting despite regularization strategies (layer freezing, dropout, SMILES augmentation). The training R² reached ~0.8 while validation plateaued around 0.55, and the model ran for 56 epochs before early stopping triggered. The final test R² of 0.531 confirms the model memorized training data rather than learning fully generalizable patterns.

**Attention analysis reveals meaningful patterns.** Despite lower overall performance, ChemBERTa's attention weights showed chemically sensible behavior — shifting focus from heteroatoms (N, O) in hydrophilic molecules to carbon backbones in lipophilic molecules, suggesting the model learned real structure-property relationships.

## Quick Start

### Installation

```bash
git clone https://github.com/Nicholas-Adrogue/adme-predictor.git
cd adme-predictor
pip install -r requirements.txt
```

> **Note:** The Therapeutics Data Commons package is installed as `pip install PyTDC` (not `tdc`). The import name is still `from tdc import ...`.

### Run the notebooks

```bash
jupyter notebook notebooks/01_baseline_model.ipynb
jupyter notebook notebooks/02_feature_engineering.ipynb
jupyter notebook notebooks/03_chemBERTa.ipynb
```

### Run training from CLI

```bash
python src/train.py --dataset lipophilicity --model xgb --features combined
```

## Project Structure

```
adme-predictor/
├── README.md
├── DESCRIPTORS.md          # Molecular descriptors reference
├── requirements.txt
├── notebooks/
│   ├── 01_baseline_model.ipynb      # EDA + Random Forest baseline
│   ├── 02_feature_engineering.ipynb  # Comparing molecular representations
│   └── 03_chemBERTa.ipynb           # Fine-tuning ChemBERTa
├── src/
│   ├── data.py          # Data loading & splitting
│   ├── featurize.py     # Molecular featurization (fingerprints, descriptors)
│   ├── train.py         # Training pipeline
│   ├── evaluate.py      # Metrics & visualization
│   └── predict.py       # Inference on new molecules
├── huggingface/
│   ├── app.py           # Gradio app for HuggingFace Spaces
│   └── requirements.txt # Space dependencies
├── docs/
│   └── index.html       # Static demo page (GitHub Pages)
├── models/              # Saved model artifacts
├── data/                # Cached datasets
└── assets/              # Plots and images
```

## Dataset

The primary dataset is **Lipophilicity (AstraZeneca)** from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/) — approximately 4,200 compounds with experimental logD values. All experiments use scaffold splitting (seed=42) to simulate realistic drug discovery conditions where models are tested on novel molecular scaffolds.

See [DESCRIPTORS.md](DESCRIPTORS.md) for a detailed explanation of each molecular property used as features.

## Tech Stack

- **Data:** Therapeutics Data Commons, RDKit, pandas
- **Classical ML:** scikit-learn, XGBoost
- **Deep Learning:** HuggingFace Transformers, PyTorch
- **Visualization:** matplotlib, seaborn
- **Deployment:** Gradio, HuggingFace Spaces

## References

- Chithrananda, S., Grand, G., & Ramsundar, B. (2020). [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
- Huang, K., et al. (2021). [Therapeutics Data Commons](https://tdcommons.ai/)
- RDKit: Open-source cheminformatics — [rdkit.org](https://www.rdkit.org/)

## Acknowledgments

- Datasets from [Therapeutics Data Commons](https://tdcommons.ai/) (Huang et al., 2021)
- Pretrained model from [ChemBERTa](https://arxiv.org/abs/2010.09885) (Chithrananda et al., 2020)
- Project scaffolding and template code developed with assistance from [Claude](https://claude.ai) (Anthropic)

## License

MIT
