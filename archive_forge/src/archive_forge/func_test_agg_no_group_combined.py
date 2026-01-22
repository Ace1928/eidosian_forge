from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_agg_no_group_combined(self):

    def assert_eq(df, gp_map, expected, expected_cols):
        res = self.to_pandas_df(self.op.group_agg(df, [], gp_map))
        assert_df_eq(res, expected, expected_cols)
    pdf = self.to_df([[0, 1, None], [0, 1, None], [0, 4, 1], [1, None, None], [None, None, None]], ['a', 'b', 'c'])
    assert_eq(pdf, {'a1': ('b,c', AggFunctionSpec('count', unique=False, dropna=False)), 'a2': ('*', AggFunctionSpec('count', unique=True, dropna=False)), 'a3': ('a', AggFunctionSpec('first', unique=True, dropna=False)), 'a4': ('b', AggFunctionSpec('max', unique=True, dropna=False))}, [[5, 4, 0, 4]], ['a1', 'a2', 'a3', 'a4'])