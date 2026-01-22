import numpy as np
import pytest
from pandas.compat import PY311
from pandas.errors import (
from pandas import (
import pandas._testing as tm
def test_methods_iloc_warn(using_copy_on_write):
    if not using_copy_on_write:
        df = DataFrame({'a': [1, 2, 3], 'b': 1})
        with tm.assert_cow_warning(match='A value'):
            df.iloc[:, 0].replace(1, 5, inplace=True)
        with tm.assert_cow_warning(match='A value'):
            df.iloc[:, 0].fillna(1, inplace=True)
        with tm.assert_cow_warning(match='A value'):
            df.iloc[:, 0].interpolate(inplace=True)
        with tm.assert_cow_warning(match='A value'):
            df.iloc[:, 0].ffill(inplace=True)
        with tm.assert_cow_warning(match='A value'):
            df.iloc[:, 0].bfill(inplace=True)