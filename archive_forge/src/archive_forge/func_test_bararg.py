import numpy as np
from numpy.testing import (assert_array_equal,
from pytest import raises as assert_raises
from scipy.special import gammaln, multigammaln
def test_bararg(self):
    assert_raises(ValueError, multigammaln, 0.5, 1.2)