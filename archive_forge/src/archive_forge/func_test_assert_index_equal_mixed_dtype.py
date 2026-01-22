import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_assert_index_equal_mixed_dtype():
    idx = Index(['foo', 'bar', 42])
    tm.assert_index_equal(idx, idx, check_order=False)