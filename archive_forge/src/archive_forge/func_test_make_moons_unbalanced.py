import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_moons_unbalanced():
    X, y = make_moons(n_samples=(7, 5))
    assert np.sum(y == 0) == 7 and np.sum(y == 1) == 5, 'Number of samples in a moon is wrong'
    assert X.shape == (12, 2), 'X shape mismatch'
    assert y.shape == (12,), 'y shape mismatch'
    with pytest.raises(ValueError, match='`n_samples` can be either an int or a two-element tuple.'):
        make_moons(n_samples=(10,))