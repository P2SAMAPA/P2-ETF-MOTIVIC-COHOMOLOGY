# Motivic Cohomology Engine for ETFs

Applies motivic cohomology and the Beilinson–Bloch regulator to ETF correlation matrices. The per‑ETF score is an “arithmetic complexity” invariant derived from the weighted product of eigenvalues.

## Features
- Three ETF universes
- Seven rolling windows (63–4536 days)
- Eigenvalue decomposition of correlation matrix
- Regulator score per ETF = ∏ λ_j^{w·v_{ij}^2}
- Optional Bloch term (log determinant of the matrix)
- Best window automatically selected (largest absolute raw score)
- Two‑tab Streamlit dashboard (auto best + manual window selection)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-motivic-cohomology-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Run training: `python train.py`
3. Launch dashboard: `streamlit run streamlit_app.py`
4. GitHub Actions runs daily.

## Interpretation

- **Motivic cohomology** is a theory from algebraic geometry (Voevodsky, Bloch, Beilinson).
- The **regulator** measures how complicated the “motive” of a correlation matrix is.
- A high regulator suggests that the ETF's return profile is arithmetically rich – potentially more predictable.
- This is the first known application of motivic cohomology to quantitative finance.

## Requirements

See `requirements.txt`.
