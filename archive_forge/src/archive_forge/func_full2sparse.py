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
def full2sparse(vec, eps=1e-09):
    """Convert a dense numpy array into the Gensim bag-of-words format.

    Parameters
    ----------
    vec : numpy.ndarray
        Dense input vector.
    eps : float
        Feature weight threshold value. Features with `abs(weight) < eps` are considered sparse and
        won't be included in the BOW result.

    Returns
    -------
    list of (int, float)
        BoW format of `vec`, with near-zero values omitted (sparse vector).

    See Also
    --------
    :func:`~gensim.matutils.sparse2full`
        Convert a document in Gensim bag-of-words format into a dense numpy array.

    """
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    return list(zip(nnz, vec.take(nnz)))