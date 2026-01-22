from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi  # gamma function utils
def ret_log_normalize_vec(vec, axis=1):
    log_max = 100.0
    if len(vec.shape) == 1:
        max_val = np.max(vec)
        log_shift = log_max - np.log(len(vec) + 1.0) - max_val
        tot = np.sum(np.exp(vec + log_shift))
        log_norm = np.log(tot) - log_shift
        vec -= log_norm
    elif axis == 1:
        max_val = np.max(vec, 1)
        log_shift = log_max - np.log(vec.shape[1] + 1.0) - max_val
        tot = np.sum(np.exp(vec + log_shift[:, np.newaxis]), 1)
        log_norm = np.log(tot) - log_shift
        vec = vec - log_norm[:, np.newaxis]
    elif axis == 0:
        k = ret_log_normalize_vec(vec.T)
        return (k[0].T, k[1])
    else:
        raise ValueError("'%s' is not a supported axis" % axis)
    return (vec, log_norm)