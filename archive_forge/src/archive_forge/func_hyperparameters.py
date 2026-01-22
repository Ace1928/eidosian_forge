import math
import warnings
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from inspect import signature
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.special import gamma, kv
from ..base import clone
from ..exceptions import ConvergenceWarning
from ..metrics.pairwise import pairwise_kernels
from ..utils.validation import _num_samples
@property
def hyperparameters(self):
    """Returns a list of all hyperparameter."""
    r = []
    for hyperparameter in self.kernel.hyperparameters:
        r.append(Hyperparameter('kernel__' + hyperparameter.name, hyperparameter.value_type, hyperparameter.bounds, hyperparameter.n_elements))
    return r