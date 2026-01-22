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
def sparse2full(doc, length):
    """Convert a document in Gensim bag-of-words format into a dense numpy array.

    Parameters
    ----------
    doc : list of (int, number)
        Document in BoW format.
    length : int
        Vector dimensionality. This cannot be inferred from the BoW, and you must supply it explicitly.
        This is typically the vocabulary size or number of topics, depending on how you created `doc`.

    Returns
    -------
    numpy.ndarray
        Dense numpy vector for `doc`.

    See Also
    --------
    :func:`~gensim.matutils.full2sparse`
        Convert dense array to gensim bag-of-words format.

    """
    result = np.zeros(length, dtype=np.float32)
    doc = ((int(id_), float(val_)) for id_, val_ in doc)
    doc = dict(doc)
    result[list(doc)] = list(doc.values())
    return result