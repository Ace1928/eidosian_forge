from __future__ import annotations
import pytest
import numpy as np
import pandas as pd
import pandas.tests.extension.base as eb
from packaging.version import Version
from datashader.datatypes import RaggedDtype, RaggedArray
from pandas.tests.extension.conftest import *  # noqa (fixture import)
def test_pandas_array_construction():
    arg = [[0, 1], [1, 2, 3, 4], None, [-1, -2]] * 2
    ra = pd.array(arg, dtype='ragged[int64]')
    expected = RaggedArray(arg, dtype='int64')
    assert_ragged_arrays_equal(ra, expected)