from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def test_LinearConstraint_combine():
    comb = LinearConstraint.combine([LinearConstraint(['a', 'b'], [1, 0]), LinearConstraint(['a', 'b'], [0, 1], [1])])
    assert comb.variable_names == ['a', 'b']
    try:
        from numpy.testing import assert_equal
    except ImportError:
        from numpy.testing.utils import assert_equal
    assert_equal(comb.coefs, [[1, 0], [0, 1]])
    assert_equal(comb.constants, [[0], [1]])
    import pytest
    pytest.raises(ValueError, LinearConstraint.combine, [])
    pytest.raises(ValueError, LinearConstraint.combine, [LinearConstraint(['a'], [1]), LinearConstraint(['b'], [1])])