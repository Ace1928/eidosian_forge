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
def test_save_and_load_csv(self):
    b = ArrayDataFrame([[6.1, 1.1], [2.1, 7.1]], 'c:double,a:double')
    path = os.path.join(self.tmpdir, 'a', 'b')
    fa.save(b, path, format_hint='csv', header=True)
    c = fa.load(path, format_hint='csv', header=True, infer_schema=True, columns=['a', 'c'], as_fugue=True)
    df_eq(c, [[1.1, 6.1], [7.1, 2.1]], 'a:double,c:double', throw=True)