import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_fail_to_catch_actual_warning(pair_different_warnings):
    expected_category, actual_category = pair_different_warnings
    match = 'Did not see expected warning of class'
    with pytest.raises(AssertionError, match=match):
        with tm.assert_produces_warning(expected_category):
            warnings.warn('warning message', actual_category)