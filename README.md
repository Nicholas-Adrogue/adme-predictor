# 🧪 ADME Property Predictor

**Predict pharmacokinetic properties of drug molecules using classical ML and fine-tuned ChemBERTa.**

Given a drug molecule as a [SMILES string](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system), this project predicts key ADME (Absorption, Distribution, Metabolism, Excretion) properties — starting with lipophilicity (logD).

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

---

## Why This Matters

ADME properties determine whether a drug candidate will actually work in the human body. Poor pharmacokinetics is one of the leading causes of drug failure in clinical trials. Predicting these properties computationally saves time and money in early-stage drug discovery.

## Project Overview

This project takes an incremental approach — building from classical ML baselines to modern transformer-based models:

| Phase | Approach | Features | Status |
|-------|----------|----------|--------|
| 1 | Random Forest / XGBoost | Morgan fingerprints | ✅ Baseline |
| 2 | Feature engineering | Multiple descriptor types | ✅ Features Examined |
| 3 | ChemBERTa fine-tuning | Learned SMILES representations | 🔄 In progress |
| 4 | Demo & deployment | Interactive web UI | ⬚ Planned |

## Quick Start

### Installation

```bash
git clone https://github.com/Nicholas-Adrogue/adme-predictor.git
cd adme-predictor
pip install -r requirements.txt
```

### Run the baseline notebook

```bash
jupyter notebook notebooks/01_baseline_model.ipynb
```

### Run training from CLI

```bash
python src/train.py --dataset lipophilicity --model rf
```

## Project Structure

```
adme-predictor/
├── README.md
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
├── docs/
│   └── index.html       # Interactive web demo
├── models/              # Saved model artifacts
├── data/                # Cached datasets
└── assets/              # Images for README & demo
```

## Datasets

All datasets are sourced from [Therapeutics Data Commons (TDC)](https://tdcommons.ai/):

- **Lipophilicity (AstraZeneca)** — ~4,200 compounds, logD regression
- **Caco-2 Permeability** — ~900 compounds, intestinal absorption
- **Solubility (AqSolDB)** — ~9,900 compounds, aqueous solubility

## Results

| Model | Dataset | RMSE | R² | MAE |
|-------|---------|------|----|-----|
| Random Forest | Lipophilicity | — | — | — |
| XGBoost | Lipophilicity | — | — | — |
| ChemBERTa | Lipophilicity | — | — | — |

*Results will be updated as each phase is completed.*

## Key Learnings

- How drug molecules are represented as strings (SMILES notation)
- Molecular fingerprints and why representation matters for ML
- Transfer learning with domain-specific language models
- The gap between classical and deep learning approaches in cheminformatics

## Tech Stack

- **Data:** Therapeutics Data Commons, RDKit, pandas
- **Classical ML:** scikit-learn, XGBoost
- **Deep Learning:** HuggingFace Transformers, PyTorch
- **Visualization:** matplotlib, seaborn
- **Demo:** HTML/CSS/JS (static, hostable anywhere)

## Important Background Information

- See [DESCRIPTORS.md](DESCRIPTORS.md) for a detailed explanation of each molecular property.


## References

- Chithrananda, S., Grand, G., & Ramsundar, B. (2020). [ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction](https://arxiv.org/abs/2010.09885)
- Huang, K., et al. (2021). [Therapeutics Data Commons](https://tdcommons.ai/)
- RDKit: Open-source cheminformatics — [rdkit.org](https://www.rdkit.org/)

## License

MIT
