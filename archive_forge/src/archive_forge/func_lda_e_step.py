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
def lda_e_step(doc_word_ids, doc_word_counts, alpha, beta, max_iter=100):
    """Performs EM-iteration on a single document for calculation of likelihood for a maximum iteration of `max_iter`.

    Parameters
    ----------
    doc_word_ids : int
        Id of corresponding words in a document.
    doc_word_counts : int
        Count of words in a single document.
    alpha : numpy.ndarray
        Lda equivalent value of alpha.
    beta : numpy.ndarray
        Lda equivalent value of beta.
    max_iter : int, optional
        Maximum number of times the expectation will be maximised.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Computed (:math:`likelihood`, :math:`\\gamma`).

    """
    gamma = np.ones(len(alpha))
    expElogtheta = np.exp(dirichlet_expectation(gamma))
    betad = beta[:, doc_word_ids]
    phinorm = np.dot(expElogtheta, betad) + 1e-100
    counts = np.array(doc_word_counts)
    for _ in range(max_iter):
        lastgamma = gamma
        gamma = alpha + expElogtheta * np.dot(counts / phinorm, betad.T)
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = np.exp(Elogtheta)
        phinorm = np.dot(expElogtheta, betad) + 1e-100
        meanchange = mean_absolute_difference(gamma, lastgamma)
        if meanchange < meanchangethresh:
            break
    likelihood = np.sum(counts * np.log(phinorm))
    likelihood += np.sum((alpha - gamma) * Elogtheta)
    likelihood += np.sum(gammaln(gamma) - gammaln(alpha))
    likelihood += gammaln(np.sum(alpha)) - gammaln(np.sum(gamma))
    return (likelihood, gamma)