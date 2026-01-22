import numbers
import operator
import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises
def test_forward_binary_methods(self):
    array = np.array([-1, 0, 1, 2])
    array_like = ArrayLike(array)
    for op in _ALL_BINARY_OPERATORS:
        expected = wrap_array_like(op(array, 1))
        actual = op(array_like, 1)
        err_msg = 'failed for operator {}'.format(op)
        _assert_equal_type_and_value(expected, actual, err_msg=err_msg)