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
@pytest.mark.filterwarnings('ignore:The default value of `n_components` will change')
def test_mbnmf_inverse_transform():
    rng = np.random.RandomState(0)
    A = np.abs(rng.randn(6, 4))
    nmf = MiniBatchNMF(random_state=rng, max_iter=500, init='nndsvdar', fresh_restarts=True)
    ft = nmf.fit_transform(A)
    A_new = nmf.inverse_transform(ft)
    assert_allclose(A, A_new, rtol=0.001, atol=0.01)