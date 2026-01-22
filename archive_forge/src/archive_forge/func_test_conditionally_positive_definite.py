import pickle
import pytest
import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats.qmc import Halton
from scipy.spatial import cKDTree
from scipy.interpolate._rbfinterp import (
from scipy.interpolate import _rbfinterp_pythran
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_conditionally_positive_definite(kernel):
    m = _NAME_TO_MIN_DEGREE.get(kernel, -1) + 1
    assert _is_conditionally_positive_definite(kernel, m)