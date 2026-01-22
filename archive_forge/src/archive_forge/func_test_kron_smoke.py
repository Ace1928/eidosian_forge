import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
@pytest.mark.parametrize('array_class', [np.asarray, np.mat])
def test_kron_smoke(self, array_class):
    a = array_class(np.ones([3, 3]))
    b = array_class(np.ones([3, 3]))
    k = array_class(np.ones([9, 9]))
    assert_array_equal(np.kron(a, b), k)