import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('check_categorical', [True, False])
@pytest.mark.parametrize('check_names', [True, False])
def test_assert_ea_index_equal_non_matching_na(check_names, check_categorical):
    idx1 = Index([1, 2], dtype='Int64')
    idx2 = Index([1, NA], dtype='Int64')
    with pytest.raises(AssertionError, match='50.0 %'):
        tm.assert_index_equal(idx1, idx2, check_names=check_names, check_categorical=check_categorical)