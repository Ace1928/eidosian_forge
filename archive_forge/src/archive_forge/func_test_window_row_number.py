from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_window_row_number(self):

    def make_func(partition_keys, order_by):
        ws = WindowSpec('', partition_keys=partition_keys, order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(''))
        return WindowFunctionSpec('row_number', False, False, ws)
    a = self.to_df([[0, 1], [0, 2]], ['a', 'b'])
    self.w_eq(a, make_func([], []), [], 'x', [[0, 1, 1], [0, 2, 2]], ['a', 'b', 'x'])
    self.w_eq(a, make_func([], [('a', True), ('b', False)]), [], 'x', [[0, 1, 2], [0, 2, 1]], ['a', 'b', 'x'])
    a = self.to_df([[0, 1], [0, None], [None, 1], [None, None]], ['a', 'b'])
    self.w_eq(a, make_func([], [('a', True), ('b', True)]), [], 'x', [[0, 1, 4], [0, None, 3], [None, 1, 2], [None, None, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func([], [('a', True, 'last'), ('b', True, 'last')]), [], 'x', [[0, 1, 1], [0, None, 2], [None, 1, 3], [None, None, 4]], ['a', 'b', 'x'])
    a = self.to_df([[0, 1], [0, 4], [2, 3]], ['a', 'b'])
    self.w_eq(a, make_func([], [('b', False)]), [], 'x', [[0, 1, 3], [0, 4, 1], [2, 3, 2]], ['a', 'b', 'x'])