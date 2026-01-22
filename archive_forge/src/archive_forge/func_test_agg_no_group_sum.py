from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_agg_no_group_sum(self):

    def assert_eq(df, gp_map, expected, expected_cols):
        res = self.to_pandas_df(self.op.group_agg(df, [], gp_map))
        assert_df_eq(res, expected, expected_cols)
    pdf = self.to_df([[0, 1], [0, 1], [1, None], [1, 5], [0, None]], ['a', 'b'])
    assert_eq(pdf, {'b1': ('b', AggFunctionSpec('sum', unique=False, dropna=False)), 'b2': ('b', AggFunctionSpec('sum', unique=True, dropna=False)), 'b3': ('b', AggFunctionSpec('sum', unique=True, dropna=True)), 'b4': ('b', AggFunctionSpec('sum', unique=False, dropna=True))}, [[7, 6, 6, 7]], ['b1', 'b2', 'b3', 'b4'])