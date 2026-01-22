import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
def test_float32_aware_assert_allclose():
    assert_allclose(np.array([1.0 + 2e-05], dtype=np.float32), 1.0)
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1.0 + 0.0002], dtype=np.float32), 1.0)
    assert_allclose(np.array([1.0 + 2e-08], dtype=np.float64), 1.0)
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1.0 + 2e-07], dtype=np.float64), 1.0)
    with pytest.raises(AssertionError):
        assert_allclose(np.array([1e-05], dtype=np.float32), 0.0)
    assert_allclose(np.array([1e-05], dtype=np.float32), 0.0, atol=2e-05)