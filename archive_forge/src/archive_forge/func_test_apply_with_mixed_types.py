from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.groupby import get_groupby_method_args
def test_apply_with_mixed_types():
    df = DataFrame({'A': 'a a b'.split(), 'B': [1, 2, 3], 'C': [4, 6, 5]})
    g = df.groupby('A', group_keys=False)
    result = g.transform(lambda x: x / x.sum())
    expected = DataFrame({'B': [1 / 3.0, 2 / 3.0, 1], 'C': [0.4, 0.6, 1.0]})
    tm.assert_frame_equal(result, expected)
    result = g.apply(lambda x: x / x.sum())
    tm.assert_frame_equal(result, expected)