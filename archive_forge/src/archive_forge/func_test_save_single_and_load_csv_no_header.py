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
def test_save_single_and_load_csv_no_header(self):
    b = ArrayDataFrame([[6.1, 1.1], [2.1, 7.1]], 'c:double,a:double')
    path = os.path.join(self.tmpdir, 'a', 'b')
    makedirs(path, exist_ok=True)
    fa.save(b, path, format_hint='csv', header=False, force_single=True)
    assert isfile(path)
    with raises(ValueError):
        c = fa.load(path, format_hint='csv', header=False, infer_schema=False, as_fugue=True)
    c = fa.load(path, format_hint='csv', header=False, infer_schema=False, columns=['c', 'a'], as_fugue=True)
    df_eq(c, [['6.1', '1.1'], ['2.1', '7.1']], 'c:str,a:str', throw=True)
    c = fa.load(path, format_hint='csv', header=False, infer_schema=True, columns=['c', 'a'], as_fugue=True)
    df_eq(c, [[6.1, 1.1], [2.1, 7.1]], 'c:double,a:double', throw=True)
    with raises(ValueError):
        c = fa.load(path, format_hint='csv', header=False, infer_schema=True, columns='c:double,a:double', as_fugue=True)
    c = fa.load(path, format_hint='csv', header=False, infer_schema=False, columns='c:double,a:str', as_fugue=True)
    df_eq(c, [[6.1, '1.1'], [2.1, '7.1']], 'c:double,a:str', throw=True)