import warnings
import pytest
from pandas.errors import (
import pandas._testing as tm
def test_ignore_extra_warning(pair_different_warnings):
    expected_category, extra_category = pair_different_warnings
    with tm.assert_produces_warning(expected_category, raise_on_extra_warnings=False):
        warnings.warn('Expected warning', expected_category)
        warnings.warn('Unexpected warning OK', extra_category)