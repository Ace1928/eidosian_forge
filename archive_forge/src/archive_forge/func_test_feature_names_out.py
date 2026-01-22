import re
import sys
from io import StringIO
import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS, LIL_CONTAINERS
from sklearn.utils.validation import assert_all_finite
@pytest.mark.parametrize('method', ['fit', 'partial_fit'])
def test_feature_names_out(method):
    """Check `get_feature_names_out` for `BernoulliRBM`."""
    n_components = 10
    rbm = BernoulliRBM(n_components=n_components)
    getattr(rbm, method)(Xdigits)
    names = rbm.get_feature_names_out()
    expected_names = [f'bernoullirbm{i}' for i in range(n_components)]
    assert_array_equal(expected_names, names)