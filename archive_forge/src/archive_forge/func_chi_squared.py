import math
import numpy as np
from tensorflow.python.ops.distributions import special_math
def chi_squared(x, bins):
    """Pearson's Chi-squared test."""
    x = np.ravel(x)
    n = len(x)
    histogram, _ = np.histogram(x, bins=bins, range=(0, 1))
    expected = n / float(bins)
    return np.sum(np.square(histogram - expected) / expected)