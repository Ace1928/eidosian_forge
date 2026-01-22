from . import matrix
from . import utils
from builtins import super
from copy import copy as shallow_copy
from future.utils import with_metaclass
from inspect import signature
from scipy import sparse
from scipy.sparse.csgraph import shortest_path
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import abc
import numbers
import numpy as np
import pickle
import pygsp
import sys
import tasklogger
import warnings
def symmetrize_kernel(self, K):
    if self.kernel_symm == '+':
        _logger.log_debug('Using addition symmetrization.')
        K = (K + K.T) / 2
    elif self.kernel_symm == '*':
        _logger.log_debug('Using multiplication symmetrization.')
        K = K.multiply(K.T)
    elif self.kernel_symm == 'mnn':
        _logger.log_debug('Using mnn symmetrization (theta = {}).'.format(self.theta))
        K = self.theta * matrix.elementwise_minimum(K, K.T) + (1 - self.theta) * matrix.elementwise_maximum(K, K.T)
    elif self.kernel_symm is None:
        _logger.log_debug('Using no symmetrization.')
        pass
    else:
        raise NotImplementedError
    return K