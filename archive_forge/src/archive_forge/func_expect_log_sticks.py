from `Wang, Paisley, Blei: "Online Variational Inference for the Hierarchical Dirichlet Process",  JMLR (2011)
import logging
import time
import warnings
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from gensim import interfaces, utils, matutils
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.models import basemodel, ldamodel
from gensim.utils import deprecated
def expect_log_sticks(sticks):
    """For stick-breaking hdp, get the :math:`\\mathbb{E}[log(sticks)]`.

    Parameters
    ----------
    sticks : numpy.ndarray
        Array of values for stick.

    Returns
    -------
    numpy.ndarray
        Computed :math:`\\mathbb{E}[log(sticks)]`.

    """
    dig_sum = psi(np.sum(sticks, 0))
    ElogW = psi(sticks[0]) - dig_sum
    Elog1_W = psi(sticks[1]) - dig_sum
    n = len(sticks[0]) + 1
    Elogsticks = np.zeros(n)
    Elogsticks[0:n - 1] = ElogW
    Elogsticks[1:] = Elogsticks[1:] + np.cumsum(Elog1_W)
    return Elogsticks