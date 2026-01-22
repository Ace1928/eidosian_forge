from inspect import signature
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.gaussian_process.kernels import (
from sklearn.metrics.pairwise import (
from sklearn.utils._testing import (
def test_kernel_operator_commutative():
    assert_almost_equal((RBF(2.0) + 1.0)(X), (1.0 + RBF(2.0))(X))
    assert_almost_equal((3.0 * RBF(2.0))(X), (RBF(2.0) * 3.0)(X))