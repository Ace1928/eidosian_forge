import logging
from itertools import chain
from copy import deepcopy
from shutil import copyfile
from os.path import isfile
from os import remove
import numpy as np  # for arrays, array broadcasting etc.
from scipy.special import gammaln  # gamma function utils
from gensim import utils
from gensim.models import LdaModel
from gensim.models.ldamodel import LdaState
from gensim.matutils import dirichlet_expectation, mean_absolute_difference
from gensim.corpora import MmCorpus
def compute_phinorm(self, expElogthetad, expElogbetad):
    """Efficiently computes the normalizing factor in phi.

        Parameters
        ----------
        expElogthetad: numpy.ndarray
            Value of variational distribution :math:`q(\\theta|\\gamma)`.
        expElogbetad: numpy.ndarray
            Value of variational distribution :math:`q(\\beta|\\lambda)`.

        Returns
        -------
        float
            Value of normalizing factor.

        """
    expElogtheta_sum = expElogthetad.sum(axis=0)
    phinorm = expElogtheta_sum.dot(expElogbetad) + 1e-100
    return phinorm