import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nth_multi_grouper(three_group):
    grouped = three_group.groupby(['A', 'B'])
    result = grouped.nth(0)
    expected = three_group.iloc[[0, 3, 4, 7]]
    tm.assert_frame_equal(result, expected)