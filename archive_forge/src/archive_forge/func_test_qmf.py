import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def test_qmf(self):
    with pytest.deprecated_call():
        assert_array_equal(wavelets.qmf([1, 1]), [1, -1])