import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_methods_iloc_getitem_item_cache_fillna(using_copy_on_write, warn_copy_on_write):
    df_orig = DataFrame({'a': [1, 2, 3], 'b': 1})
    df = df_orig.copy()
    ser = df.iloc[:, 0]
    ser.fillna(1, inplace=True)
    df = df_orig.copy()
    ser = df.copy()['a']
    ser.fillna(1, inplace=True)
    df = df_orig.copy()
    df['a']
    ser = df.iloc[:, 0]
    ser.fillna(1, inplace=True)
    df = df_orig.copy()
    df['a']
    ser = df['a']
    ser.fillna(1, inplace=True)
    df = df_orig.copy()
    df['a']
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['a'].fillna(1, inplace=True)
    else:
        with tm.assert_cow_warning(match='A value'):
            df['a'].fillna(1, inplace=True)
    df = df_orig.copy()
    ser = df['a']
    if using_copy_on_write:
        with tm.raises_chained_assignment_error():
            df['a'].fillna(1, inplace=True)
    else:
        with tm.assert_cow_warning(warn_copy_on_write, match='A value'):
            df['a'].fillna(1, inplace=True)