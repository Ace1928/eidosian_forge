from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def test_agg_count(self):

    def assert_eq(df, gp_keys, gp_map, expected, expected_cols):
        res = self.to_pandas_df(self.op.group_agg(df, gp_keys, gp_map))
        assert_df_eq(res, expected, expected_cols)
    pdf = self.to_df([[0, 1], [0, 1], [0, 4], [1, None], [0, None]], ['a', 'b'])
    assert_eq(pdf, ['a'], {'b1': ('b', AggFunctionSpec('count', unique=False, dropna=False)), 'b2': ('b', AggFunctionSpec('count', unique=True, dropna=False)), 'b3': ('b', AggFunctionSpec('count', unique=False, dropna=True)), 'b4': ('b', AggFunctionSpec('count', unique=True, dropna=True))}, [[4, 3, 3, 2], [1, 1, 0, 0]], ['b1', 'b2', 'b3', 'b4'])
    assert_eq(pdf, ['a'], {'b1': ('*', AggFunctionSpec('count', unique=False, dropna=False)), 'b2': ('*', AggFunctionSpec('count', unique=True, dropna=False))}, [[4, 3], [1, 1]], ['b1', 'b2'])
    assert_eq(pdf, ['b'], {'a1': ('*', AggFunctionSpec('count', unique=False, dropna=False)), 'a2': ('*', AggFunctionSpec('count', unique=True, dropna=False))}, [[2, 2], [2, 1], [1, 1]], ['a1', 'a2'])
    pdf = self.to_df([[0, 1, None], [0, 1, None], [0, 4, 1], [1, None, None], [None, None, None]], ['a', 'b', 'c'])
    assert_eq(pdf, ['a'], {'a1': ('b,c', AggFunctionSpec('count', unique=False, dropna=False)), 'a2': ('b,c', AggFunctionSpec('count', unique=True, dropna=False))}, [[3, 2], [1, 1], [1, 1]], ['a1', 'a2'])