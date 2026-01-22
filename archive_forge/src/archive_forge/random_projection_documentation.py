import warnings
from abc import ABCMeta, abstractmethod
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from .base import (
from .exceptions import DataDimensionalityWarning
from .utils import check_random_state
from .utils._param_validation import Interval, StrOptions, validate_params
from .utils.extmath import safe_sparse_dot
from .utils.random import sample_without_replacement
from .utils.validation import check_array, check_is_fitted
Project the data by using matrix product with the random matrix.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input data to project into a smaller dimensional space.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Projected array. It is a sparse matrix only when the input is sparse and
            `dense_output = False`.
        