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
def test_nmf_non_negative_factorization_n_components_auto():
    rng = np.random.RandomState(0)
    X = rng.random_sample((6, 5))
    W_init = rng.random_sample((6, 2))
    H_init = rng.random_sample((2, 5))
    W, H, _ = non_negative_factorization(X, W=W_init, H=H_init, init='custom', n_components='auto')
    assert H.shape == H_init.shape
    assert W.shape == W_init.shape