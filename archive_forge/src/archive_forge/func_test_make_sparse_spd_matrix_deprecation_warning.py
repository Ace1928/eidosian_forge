import re
from collections import defaultdict
from functools import partial
import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.datasets import (
from sklearn.utils._testing import (
from sklearn.utils.validation import assert_all_finite
def test_make_sparse_spd_matrix_deprecation_warning():
    """Check the message for future deprecation."""
    warn_msg = 'dim was deprecated in version 1.4'
    with pytest.warns(FutureWarning, match=warn_msg):
        make_sparse_spd_matrix(dim=1)
    error_msg = '`dim` and `n_dim` cannot be both specified'
    with pytest.raises(ValueError, match=error_msg):
        make_sparse_spd_matrix(dim=1, n_dim=1)
    X = make_sparse_spd_matrix()
    assert X.shape[1] == 1