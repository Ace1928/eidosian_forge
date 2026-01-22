import time
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy.special import expit  # logistic function
from ..base import (
from ..utils import check_random_state, gen_even_slices
from ..utils._param_validation import Interval
from ..utils.extmath import safe_sparse_dot
from ..utils.validation import check_is_fitted
def gibbs(self, v):
    """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : ndarray of shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : ndarray of shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
    check_is_fitted(self)
    if not hasattr(self, 'random_state_'):
        self.random_state_ = check_random_state(self.random_state)
    h_ = self._sample_hiddens(v, self.random_state_)
    v_ = self._sample_visibles(h_, self.random_state_)
    return v_