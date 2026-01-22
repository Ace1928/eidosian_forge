import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_spd_matrix():
    X = make_spd_matrix(n_dim=5, random_state=0)
    assert X.shape == (5, 5), 'X shape mismatch'
    assert_array_almost_equal(X, X.T)
    from numpy.linalg import eig
    eigenvalues, _ = eig(X)
    assert np.all(eigenvalues > 0), 'X is not positive-definite'