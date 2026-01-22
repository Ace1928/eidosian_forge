import numpy as np
import pytest
from pandas.compat import PY311
from pandas.core.dtypes.dtypes import DatetimeTZDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_set_levels_with_iterable():
    sizes = [1, 2, 3]
    colors = ['black'] * 3
    index = MultiIndex.from_arrays([sizes, colors], names=['size', 'color'])
    result = index.set_levels(map(int, ['3', '2', '1']), level='size')
    expected_sizes = [3, 2, 1]
    expected = MultiIndex.from_arrays([expected_sizes, colors], names=['size', 'color'])
    tm.assert_index_equal(result, expected)