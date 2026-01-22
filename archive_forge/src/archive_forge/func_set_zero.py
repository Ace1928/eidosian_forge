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
def set_zero(self):
    """Fill the sticks and beta array with 0 scalar value."""
    self.m_var_sticks_ss.fill(0.0)
    self.m_var_beta_ss.fill(0.0)