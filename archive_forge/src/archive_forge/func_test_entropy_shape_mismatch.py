import numpy as np
from numpy.testing import assert_equal, assert_allclose
from numpy.testing import (assert_, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.stats as stats
def test_entropy_shape_mismatch(self):
    x = np.random.rand(10, 1, 12)
    y = np.random.rand(11, 2)
    message = 'Array shapes are incompatible for broadcasting.'
    with pytest.raises(ValueError, match=message):
        stats.entropy(x, y)