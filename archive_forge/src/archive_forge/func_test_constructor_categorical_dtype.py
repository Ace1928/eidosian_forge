from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
def test_constructor_categorical_dtype(self):
    result = Series(['a', 'b'], dtype=CategoricalDtype(['a', 'b', 'c'], ordered=True))
    assert isinstance(result.dtype, CategoricalDtype)
    tm.assert_index_equal(result.cat.categories, Index(['a', 'b', 'c']))
    assert result.cat.ordered
    result = Series(['a', 'b'], dtype=CategoricalDtype(['b', 'a']))
    assert isinstance(result.dtype, CategoricalDtype)
    tm.assert_index_equal(result.cat.categories, Index(['b', 'a']))
    assert result.cat.ordered is False
    result = Series('a', index=[0, 1], dtype=CategoricalDtype(['a', 'b'], ordered=True))
    expected = Series(['a', 'a'], index=[0, 1], dtype=CategoricalDtype(['a', 'b'], ordered=True))
    tm.assert_series_equal(result, expected)