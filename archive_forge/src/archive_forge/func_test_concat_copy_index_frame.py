from copy import deepcopy
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_copy_index_frame(self, axis, using_copy_on_write):
    df = DataFrame([[1, 2], [3, 4]], columns=['a', 'b'])
    comb = concat([df, df], axis=axis, copy=True)
    if not using_copy_on_write:
        assert not comb.index.is_(df.index)
        assert not comb.columns.is_(df.columns)
    elif axis in [0, 'index']:
        assert not comb.index.is_(df.index)
        assert comb.columns.is_(df.columns)
    elif axis in [1, 'columns']:
        assert comb.index.is_(df.index)
        assert not comb.columns.is_(df.columns)