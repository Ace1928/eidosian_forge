from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('result_1, duplicate_item, expected_1', [[Series({1: 12, 2: [1, 2, 2, 3]}), Series({1: 313}), Series({1: 12}, dtype=object)], [Series({1: [1, 2, 3], 2: [1, 2, 2, 3]}), Series({1: [1, 2, 3]}), Series({1: [1, 2, 3]})]])
def test_getitem_with_duplicates_indices(result_1, duplicate_item, expected_1):
    result = result_1._append(duplicate_item)
    expected = expected_1._append(duplicate_item)
    tm.assert_series_equal(result[1], expected)
    assert result[2] == result_1[2]