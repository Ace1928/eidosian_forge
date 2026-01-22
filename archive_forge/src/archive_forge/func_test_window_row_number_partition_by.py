from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_window_row_number_partition_by(self):

    def make_func(partition_keys, order_by):
        ws = WindowSpec('', partition_keys=partition_keys, order_by=OrderBySpec(*order_by), windowframe=make_windowframe_spec(''))
        return WindowFunctionSpec('row_number', False, False, ws)
    a = self.to_df([[0, 1], [0, None], [None, 1], [None, None]], ['a', 'b'])
    self.w_eq(a, make_func(['a'], [('a', True), ('b', True)]), [], 'x', [[0, 1, 2], [0, None, 1], [None, 1, 2], [None, None, 1]], ['a', 'b', 'x'])