import pytest
import numpy as np
from numpy.core import (
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
from numpy.testing import (
@pytest.mark.skipif(IS_PYPY, reason='PYPY handles sq_concat, nb_add differently than cpython')
def test_operator_concat(self):
    import operator
    a = array([1, 2])
    b = array([3, 4])
    n = [1, 2]
    res = array([1, 2, 3, 4])
    assert_raises(TypeError, operator.concat, a, b)
    assert_raises(TypeError, operator.concat, a, n)
    assert_raises(TypeError, operator.concat, n, a)
    assert_raises(TypeError, operator.concat, a, 1)
    assert_raises(TypeError, operator.concat, 1, a)