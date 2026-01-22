import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_intersection_equal_different_names():
    mi1 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['c', 'b'])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]], names=['a', 'b'])
    result = mi1.intersection(mi2)
    expected = MultiIndex.from_arrays([[1, 2], [3, 4]], names=[None, 'b'])
    tm.assert_index_equal(result, expected)