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
def test_nmf_n_components_auto_no_h_update():
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    H_true = rng.random_sample((2, 5))
    W, H, _ = non_negative_factorization(X, H=H_true, n_components='auto', update_H=False)
    assert_allclose(H, H_true)
    assert W.shape == (X.shape[0], H_true.shape[0])