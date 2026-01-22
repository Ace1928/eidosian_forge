from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_window_ranks_partition_by(self):

    def make_func(func, partition_keys, order_by):
        ws = WindowSpec('', partition_keys=partition_keys, order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(''))
        return WindowFunctionSpec(func, False, False, ws)
    a = self.to_df([[0, 1], [0, 1], [None, 1], [0, None]], ['a', 'b'])
    self.w_eq(a, make_func('rank', ['a'], ['b']), [], 'x', [[0, 1, 2], [0, 1, 2], [None, 1, 1], [0, None, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('rank', ['a'], [('b', False)]), [], 'x', [[0, 1, 1], [0, 1, 1], [None, 1, 1], [0, None, 3]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('dense_rank', ['a'], ['b']), [], 'x', [[0, 1, 2], [0, 1, 2], [None, 1, 1], [0, None, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('dense_rank', ['a'], [('b', False)]), [], 'x', [[0, 1, 1], [0, 1, 1], [None, 1, 1], [0, None, 2]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('percentile_rank', ['a'], ['b']), [], 'x', [[0, 1, 0.5], [0, 1, 0.5], [None, 1, 0.0], [0, None, 0.0]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('percentile_rank', ['a'], [('b', False)]), [], 'x', [[0, 1, 0.0], [0, 1, 0.0], [None, 1, 0.0], [0, None, 1.0]], ['a', 'b', 'x'])