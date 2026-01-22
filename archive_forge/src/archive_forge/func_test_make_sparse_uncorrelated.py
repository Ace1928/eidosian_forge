import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_sparse_uncorrelated():
    X, y = make_sparse_uncorrelated(n_samples=5, n_features=10, random_state=0)
    assert X.shape == (5, 10), 'X shape mismatch'
    assert y.shape == (5,), 'y shape mismatch'