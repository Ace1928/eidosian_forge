import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import svds
from scipy.optimize import fminbound
import warnings
from statsmodels.tools.tools import Bunch
from statsmodels.tools.sm_exceptions import (
def clip_evals(x, value=0):
    evals, evecs = np.linalg.eigh(x)
    clipped = np.any(evals < value)
    x_new = np.dot(evecs * np.maximum(evals, value), evecs.T)
    return (x_new, clipped)