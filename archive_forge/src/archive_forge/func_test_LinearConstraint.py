from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def test_LinearConstraint():
    try:
        from numpy.testing import assert_equal
    except ImportError:
        from numpy.testing.utils import assert_equal
    lc = LinearConstraint(['foo', 'bar'], [1, 1])
    assert lc.variable_names == ['foo', 'bar']
    assert_equal(lc.coefs, [[1, 1]])
    assert_equal(lc.constants, [[0]])
    lc = LinearConstraint(['foo', 'bar'], [[1, 1], [2, 3]], [10, 20])
    assert_equal(lc.coefs, [[1, 1], [2, 3]])
    assert_equal(lc.constants, [[10], [20]])
    assert lc.coefs.dtype == np.dtype(float)
    assert lc.constants.dtype == np.dtype(float)
    lc = LinearConstraint(['a'], [[0]])
    assert_equal(lc.coefs, [[0]])
    import pytest
    pytest.raises(ValueError, LinearConstraint, ['a'], [[1, 2]])
    pytest.raises(ValueError, LinearConstraint, ['a'], [[[1]]])
    pytest.raises(ValueError, LinearConstraint, ['a'], [[1, 2]], [3, 4])
    pytest.raises(ValueError, LinearConstraint, ['a', 'b'], [[1, 2]], [3, 4])
    pytest.raises(ValueError, LinearConstraint, ['a'], [[1]], [[]])
    pytest.raises(ValueError, LinearConstraint, ['a', 'b'], [])
    pytest.raises(ValueError, LinearConstraint, ['a', 'b'], np.zeros((0, 2)))
    assert_no_pickling(lc)