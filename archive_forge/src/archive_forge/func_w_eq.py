from typing import Any
import pandas as pd
from qpd.dataframe import Column, DataFrame
from qpd.specs import (
from qpd_test.tests_base import TestsBase
from qpd_test.utils import assert_df_eq
from pytest import raises
def w_eq(self, df, func, cols, dest, expected, expected_cols):
    cols = [x if isinstance(x, ArgumentSpec) else ArgumentSpec(False, x) for x in cols]
    res = self.to_pandas_df(self.op.window(df, func, cols, dest))
    assert_df_eq(res, expected, expected_cols)