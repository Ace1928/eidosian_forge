import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_almost_equal_dicts():
    _assert_almost_equal_both({'a': 1, 'b': 2}, {'a': 1, 'b': 2})