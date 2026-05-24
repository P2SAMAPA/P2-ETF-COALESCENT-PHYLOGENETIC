# Coalescent Phylogenetic Model for ETFs

Applies the Kingman coalescent (a population genetics model) to ETF return trajectories. Builds a phylogenetic tree using UPGMA from a distance matrix (1 - |correlation|). The **divergence score** for each ETF is the total branch length from the root to its leaf – a measure of historical uniqueness.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63, 252, 504, 1008, 2016, 4032, 4536 days) – 5040 replaced with 4536
- Distance metric: correlation or Euclidean
- Tree building: UPGMA (average linkage)
- Score = root-to-leaf depth (normalised)
- Best window selected automatically (largest absolute raw signal)
- Results stored in Hugging Face: `P2SAMAPA/p2-etf-coalescent-results`
- Streamlit dashboard with refresh button

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- The coalescent tree groups ETFs by the similarity of their price evolution.
- ETF with high divergence score has a long evolutionary branch – it is structurally different from the rest, potentially offering diversification or alpha.
- Entirely novel application in finance (first known implementation of Kingman coalescent on ETFs).

## Requirements

See `requirements.txt`.
