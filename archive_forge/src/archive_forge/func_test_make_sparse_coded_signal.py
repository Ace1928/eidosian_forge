import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_sparse_coded_signal():
    Y, D, X = make_sparse_coded_signal(n_samples=5, n_components=8, n_features=10, n_nonzero_coefs=3, random_state=0)
    assert Y.shape == (5, 10), 'Y shape mismatch'
    assert D.shape == (8, 10), 'D shape mismatch'
    assert X.shape == (5, 8), 'X shape mismatch'
    for row in X:
        assert len(np.flatnonzero(row)) == 3, 'Non-zero coefs mismatch'
    assert_allclose(Y, X @ D)
    assert_allclose(np.sqrt((D ** 2).sum(axis=1)), np.ones(D.shape[0]))