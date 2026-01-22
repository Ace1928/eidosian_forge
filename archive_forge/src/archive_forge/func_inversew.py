from scipy import stats
from scipy.stats import distributions
import numpy as np
def inversew(x):
    return 1.0 / (1 + mux + x * stdx)