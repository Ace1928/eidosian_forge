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
def test_map_with_special_values(self):

    def with_nat(cursor, data):
        df = data.as_pandas()
        df['nat'] = pd.NaT
        schema = data.schema + 'nat:datetime'
        return PandasDataFrame(df, schema)
    e = self.engine
    o = ArrayDataFrame([[1, None, 1], [1, None, 0], [None, None, 2]], 'a:double,b:double,c:int')
    c = e.map_engine.map_dataframe(o, select_top, o.schema, PartitionSpec(by=['a', 'b'], presort='c'))
    df_eq(c, [[1, None, 0], [None, None, 2]], 'a:double,b:double,c:int', throw=True)
    dt = datetime.now()
    o = ArrayDataFrame([[dt, 2, 1], [None, 2, None], [None, 1, None], [dt, 5, 1], [None, 4, None]], 'a:datetime,b:int,c:double')
    c = e.map_engine.map_dataframe(o, select_top, o.schema, PartitionSpec(by=['a', 'c'], presort='b DESC'))
    df_eq(c, [[None, 4, None], [dt, 5, 1]], 'a:datetime,b:int,c:double', throw=True)
    d = e.map_engine.map_dataframe(c, with_nat, 'a:datetime,b:int,c:double,nat:datetime', PartitionSpec())
    df_eq(d, [[None, 4, None, None], [dt, 5, 1, None]], 'a:datetime,b:int,c:double,nat:datetime', throw=True)
    o = ArrayDataFrame([[dt, [1, 2]]], 'a:datetime,b:[int]')
    c = e.map_engine.map_dataframe(o, select_top, o.schema, PartitionSpec(by=['a']))
    df_eq(c, o, check_order=True, throw=True)