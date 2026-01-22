import logging
import numbers
import os
import time
from collections import defaultdict
import numpy as np
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
from gensim import interfaces, utils, matutils
from gensim.matutils import (
from gensim.models import basemodel, CoherenceModel
from gensim.models.callbacks import Callback
def update_dir_prior(prior, N, logphat, rho):
    """Update a given prior using Newton's method, described in
    `J. Huang: "Maximum Likelihood Estimation of Dirichlet Distribution Parameters"
    <http://jonathan-huang.org/research/dirichlet/dirichlet.pdf>`_.

    Parameters
    ----------
    prior : list of float
        The prior for each possible outcome at the previous iteration (to be updated).
    N : int
        Number of observations.
    logphat : list of float
        Log probabilities for the current estimation, also called "observed sufficient statistics".
    rho : float
        Learning rate.

    Returns
    -------
    list of float
        The updated prior.

    """
    gradf = N * (psi(np.sum(prior)) - psi(prior) + logphat)
    c = N * polygamma(1, np.sum(prior))
    q = -N * polygamma(1, prior)
    b = np.sum(gradf / q) / (1 / c + np.sum(1 / q))
    dprior = -(gradf - b) / q
    updated_prior = rho * dprior + prior
    if all(updated_prior > 0):
        prior = updated_prior
    else:
        logger.warning('updated prior is not positive')
    return prior