import operator
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_logical_operators(self):

    def _check_bin_op(op):
        result = op(df1, df2)
        expected = DataFrame(op(df1.values, df2.values), index=df1.index, columns=df1.columns)
        assert result.values.dtype == np.bool_
        tm.assert_frame_equal(result, expected)

    def _check_unary_op(op):
        result = op(df1)
        expected = DataFrame(op(df1.values), index=df1.index, columns=df1.columns)
        assert result.values.dtype == np.bool_
        tm.assert_frame_equal(result, expected)
    df1 = {'a': {'a': True, 'b': False, 'c': False, 'd': True, 'e': True}, 'b': {'a': False, 'b': True, 'c': False, 'd': False, 'e': False}, 'c': {'a': False, 'b': False, 'c': True, 'd': False, 'e': False}, 'd': {'a': True, 'b': False, 'c': False, 'd': True, 'e': True}, 'e': {'a': True, 'b': False, 'c': False, 'd': True, 'e': True}}
    df2 = {'a': {'a': True, 'b': False, 'c': True, 'd': False, 'e': False}, 'b': {'a': False, 'b': True, 'c': False, 'd': False, 'e': False}, 'c': {'a': True, 'b': False, 'c': True, 'd': False, 'e': False}, 'd': {'a': False, 'b': False, 'c': False, 'd': True, 'e': False}, 'e': {'a': False, 'b': False, 'c': False, 'd': False, 'e': True}}
    df1 = DataFrame(df1)
    df2 = DataFrame(df2)
    _check_bin_op(operator.and_)
    _check_bin_op(operator.or_)
    _check_bin_op(operator.xor)
    _check_unary_op(operator.inv)