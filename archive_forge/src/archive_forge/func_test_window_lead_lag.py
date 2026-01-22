from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_window_lead_lag(self):

    def make_func(func, partition_keys, order_by):
        ws = WindowSpec('', partition_keys=partition_keys, order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(''))
        return WindowFunctionSpec(func, False, False, ws)
    a = self.to_df([[0, 1], [0, 1], [None, 1], [0, None]], ['a', 'b'])
    self.w_eq(a, make_func('lag', [], []), [col('b'), 1], 'x', [[0, 1, None], [0, 1, 1], [None, 1, 1], [0, None, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('lag', [], []), [col('b'), 1, 100], 'x', [[0, 1, 100], [0, 1, 1], [None, 1, 1], [0, None, 1]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('lead', [], []), [col('b'), 1], 'x', [[0, 1, 1], [0, 1, 1], [None, 1, None], [0, None, None]], ['a', 'b', 'x'])
    self.w_eq(a, make_func('lead', [], []), [col('b'), 1, 100], 'x', [[0, 1, 1], [0, 1, 1], [None, 1, None], [0, None, 100]], ['a', 'b', 'x'])