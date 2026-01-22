import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
def test_apply_map_same_length_inference_bug():
    s = Series([1, 2])

    def f(x):
        return (x, x + 1)
    result = s.apply(f, by_row='compat')
    expected = s.map(f)
    tm.assert_series_equal(result, expected)