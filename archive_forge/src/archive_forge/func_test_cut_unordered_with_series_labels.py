import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_unordered_with_series_labels():
    ser = Series([1, 2, 3, 4, 5])
    bins = Series([0, 2, 4, 6])
    labels = Series(['a', 'b', 'c'])
    result = cut(ser, bins=bins, labels=labels, ordered=False)
    expected = Series(['a', 'a', 'b', 'b', 'c'], dtype='category')
    tm.assert_series_equal(result, expected)