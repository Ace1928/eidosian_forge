import re
import sys
import warnings
from io import StringIO
import numpy as np
import pytest
from scipy import linalg
from sklearn.base import clone
from sklearn.decomposition import NMF, MiniBatchNMF, non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import (
from sklearn.utils.extmath import squared_norm
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_nmf_w_h_not_used_warning():
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    W_init = rng.random_sample((6, 2))
    H_init = rng.random_sample((2, 5))
    with pytest.warns(RuntimeWarning, match="When init!='custom', provided W or H are ignored"):
        non_negative_factorization(X, H=H_init, update_H=True, n_components='auto')
    with pytest.warns(RuntimeWarning, match="When init!='custom', provided W or H are ignored"):
        non_negative_factorization(X, W=W_init, H=H_init, update_H=True, n_components='auto')
    with pytest.warns(RuntimeWarning, match='When update_H=False, the provided initial W is not used.'):
        non_negative_factorization(X, W=W_init, H=H_init, update_H=False, n_components='auto')