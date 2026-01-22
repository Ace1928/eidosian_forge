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
def test__join_with_null_keys(self):
    e = self.engine
    a = fa.as_fugue_engine_df(e, [[1, 2, 3], [4, None, 6]], 'a:double,b:double,c:int')
    b = fa.as_fugue_engine_df(e, [[1, 2, 33], [4, None, 63]], 'a:double,b:double,d:int')
    c = fa.join(a, b, how='INNER')
    df_eq(c, [[1, 2, 3, 33]], 'a:double,b:double,c:int,d:int', throw=True)