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
@property
def kernel_degree(self):
    """Weighted degree vector (cached)

        Return or calculate the degree vector from the affinity matrix

        Returns
        -------

        degrees : array-like, shape=[n_samples]
            Row sums of graph kernel
        """
    try:
        return self._kernel_degree
    except AttributeError:
        self._kernel_degree = matrix.to_array(self.kernel.sum(axis=1)).reshape(-1, 1)
        return self._kernel_degree