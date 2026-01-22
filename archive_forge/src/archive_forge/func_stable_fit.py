import numba
import numpy as np
@numba.jit(nopython=True, parallel=True)
def stable_fit(X, y, threshold=3):
    min_obs = int(X.shape[1] * 1.5)
    beta = np.zeros((X.shape[1], y.shape[1]), dtype=np.float64)
    residuals = np.full_like(y, np.nan)
    stable = np.empty(y.shape[1])
    for idx in numba.prange(y.shape[1]):
        y_sub = y[:, idx]
        isna = np.isnan(y_sub)
        X_sub = X[~isna]
        y_sub = y_sub[~isna]
        is_stable = False
        for jdx in range(len(y_sub), min_obs - 1, -2):
            y_ = y_sub[-jdx:]
            X_ = X_sub[-jdx:]
            beta_sub = np.linalg.solve(np.dot(X_.T, X_), np.dot(X_.T, y_))
            resid_sub = np.dot(X_, beta_sub) - y_
            rmse = np.sqrt(np.mean(resid_sub ** 2))
            first = np.fabs(resid_sub[0]) / rmse < threshold
            last = np.fabs(resid_sub[-1]) / rmse < threshold
            is_stable = first & last
            if is_stable:
                break
        beta[:, idx] = beta_sub
        residuals[-jdx:, idx] = resid_sub
        stable[idx] = is_stable
    return (beta, residuals, stable.astype(np.bool_))