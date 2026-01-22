from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def pval_kstest_approx(D, N):
    pval_two = distributions.kstwobign.sf(D * np.sqrt(N))
    if N > 2666 or pval_two > 0.8 - N * 0.3 / 1000.0:
        return (D, distributions.kstwobign.sf(D * np.sqrt(N)), np.nan)
    else:
        return (D, distributions.ksone.sf(D, N) * 2, np.nan)