import copy
import os
import pickle
from datetime import datetime
from unittest import TestCase
import pandas as pd
import pytest
from pytest import raises
from triad.exceptions import InvalidOperationError
from triad.utils.io import isfile, makedirs, touch
import fugue.api as fa
import fugue.column.functions as ff
from fugue import (
from fugue.column import all_cols, col, lit
from fugue.dataframe.utils import _df_eq as df_eq
from fugue.execution.native_execution_engine import NativeExecutionEngine
def test_engine_api(self):
    with fa.engine_context(self.engine):
        df1 = fa.as_fugue_df([[0, 1], [2, 3]], schema='a:long,b:long')
        df1 = fa.repartition(df1, {'num': 2})
        df1 = fa.get_native_as_df(fa.broadcast(df1))
        df2 = pd.DataFrame([[0, 1], [2, 3]], columns=['a', 'b'])
        df3 = fa.union(df1, df2, as_fugue=False)
        assert fa.is_df(df3) and (not isinstance(df3, DataFrame))
        df4 = fa.union(df1, df2, as_fugue=True)
        assert isinstance(df4, DataFrame)
        df_eq(df4, fa.as_pandas(df3), throw=True)