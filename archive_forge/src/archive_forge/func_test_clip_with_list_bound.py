import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_clip_with_list_bound(self):
    df = DataFrame([1, 5])
    expected = DataFrame([3, 5])
    result = df.clip([3])
    tm.assert_frame_equal(result, expected)
    expected = DataFrame([1, 3])
    result = df.clip(upper=[3])
    tm.assert_frame_equal(result, expected)