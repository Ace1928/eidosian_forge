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
@pytest.mark.parametrize(['Estimator', 'solver'], [[NMF, {'solver': 'cd'}], [NMF, {'solver': 'mu'}], [MiniBatchNMF, {}]])
def test_nmf_fit_close(Estimator, solver):
    rng = np.random.mtrand.RandomState(42)
    pnmf = Estimator(5, init='nndsvdar', random_state=0, max_iter=600, **solver)
    X = np.abs(rng.randn(6, 5))
    assert pnmf.fit(X).reconstruction_err_ < 0.1