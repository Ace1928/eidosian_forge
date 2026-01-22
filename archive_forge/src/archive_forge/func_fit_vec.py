from statsmodels.compat.python import lmap
import numpy as np
from scipy.stats import distributions
from statsmodels.tools.decorators import cache_readonly
from scipy.special import kolmogorov as ksprob
def fit_vec(self, x, axis=0):
    return (x.mean(axis), x.std(axis))