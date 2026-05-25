import numpy as np
from scipy.linalg import eigh

def motivic_regulator(returns, weight=2.0, use_bloch=True):
    """
    Compute per‑ETF Beilinson–Bloch regulator approximation.
    Steps:
      1. Compute correlation matrix C.
      2. Eigen decomposition: C = V Λ V^T.
      3. For each ETF i, form a diagonal "regulator matrix" R_i = diag( λ_j^{weight} * v_{ij}^2 ).
      4. The regulator value for ETF i is the log determinant of R_i (Bloch term) + trace of something.
      5. Return a score = det(R_i)^{1/weight} + (if use_bloch) log(det(R_i)).
    Simplified: score_i = exp(∑_j weight * log(λ_j) * v_{ij}^2) = ∏_j λ_j^{weight * v_{ij}^2}.
    """
    returns_clean = returns.dropna()
    n = returns_clean.shape[1]
    if n < 2:
        return {t: 0.0 for t in returns_clean.columns}
    corr = returns_clean.corr().values
    eigvals, eigvecs = eigh(corr)
    # Ensure positivity
    eigvals = np.maximum(eigvals, 1e-12)
    scores = np.zeros(n)
    for i in range(n):
        log_score = 0.0
        for j in range(n):
            log_score += weight * np.log(eigvals[j]) * (eigvecs[i, j]**2)
        scores[i] = np.exp(log_score)
        if use_bloch:
            # Add a correction: log determinant of the matrix (positive definite)
            scores[i] += np.log(np.prod(eigvals) ** weight)
    tickers = returns_clean.columns
    return {ticker: scores[j] for j, ticker in enumerate(tickers)}
