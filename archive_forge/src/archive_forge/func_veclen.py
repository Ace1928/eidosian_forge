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
def veclen(vec):
    """Calculate L2 (euclidean) length of a vector.

    Parameters
    ----------
    vec : list of (int, number)
        Input vector in sparse bag-of-words format.

    Returns
    -------
    float
        Length of `vec`.

    """
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum((val ** 2 for _, val in vec)))
    assert length > 0.0, 'sparse documents must not contain any explicit zero entries'
    return length