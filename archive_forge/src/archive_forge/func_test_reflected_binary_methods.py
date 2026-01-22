import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_reflected_binary_methods(self):
    for op in _ALL_BINARY_OPERATORS:
        expected = wrap_array_like(op(2, 1))
        actual = op(2, ArrayLike(1))
        err_msg = 'failed for operator {}'.format(op)
        _assert_equal_type_and_value(expected, actual, err_msg=err_msg)