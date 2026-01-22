import pytest
from pandas import (
import pandas._testing as tm
def test_join_overlapping_interval_to_another_intervalindex(interval_index):
    flipped_interval_index = interval_index[::-1]
    result = interval_index.join(flipped_interval_index)
    tm.assert_index_equal(result, interval_index)