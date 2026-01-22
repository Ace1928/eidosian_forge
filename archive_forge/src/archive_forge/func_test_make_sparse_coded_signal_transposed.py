import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
@ignore_warnings(category=FutureWarning)
def test_make_sparse_coded_signal_transposed():
    Y, D, X = make_sparse_coded_signal(n_samples=5, n_components=8, n_features=10, n_nonzero_coefs=3, random_state=0, data_transposed=True)
    assert Y.shape == (10, 5), 'Y shape mismatch'
    assert D.shape == (10, 8), 'D shape mismatch'
    assert X.shape == (8, 5), 'X shape mismatch'
    for col in X.T:
        assert len(np.flatnonzero(col)) == 3, 'Non-zero coefs mismatch'
    assert_allclose(Y, D @ X)
    assert_allclose(np.sqrt((D ** 2).sum(axis=0)), np.ones(D.shape[1]))