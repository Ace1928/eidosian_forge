import re
import numpy as np
import pytest
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_drop_with_non_unique_multiindex(self):
    mi = MultiIndex.from_arrays([['x', 'y', 'x'], ['i', 'j', 'i']])
    df = DataFrame([1, 2, 3], index=mi)
    result = df.drop(index='x')
    expected = DataFrame([2], index=MultiIndex.from_arrays([['y'], ['j']]))
    tm.assert_frame_equal(result, expected)