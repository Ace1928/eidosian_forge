from statsmodels.compat.python import lzip, lmap
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import logsumexp as sp_logsumexp
def bitstonats(X):
    """
    Converts from bits to nats
    """
    return logbasechange(2, np.e) * X