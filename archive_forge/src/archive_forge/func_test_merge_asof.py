import contextlib
import numpy as np
import pandas
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import StorageFormat
from modin.pandas.io import to_pandas
from modin.pandas.testing import assert_frame_equal
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
from .utils import (
@pytest.mark.parametrize('right_index', [None, [0] * 5], ids=['default', 'non_unique'])
def test_merge_asof(right_index):
    left = pd.DataFrame({'a': [1, 5, 10], 'left_val': ['a', 'b', 'c']})
    right = pd.DataFrame({'a': [1, 2, 3, 6, 7], 'right_val': [1, 2, 3, 6, 7]}, index=right_index)
    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on='a')
        assert isinstance(df, pd.DataFrame)
    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on='a', allow_exact_matches=False)
        assert isinstance(df, pd.DataFrame)
    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on='a', direction='forward')
        assert isinstance(df, pd.DataFrame)
    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, on='a', direction='nearest')
        assert isinstance(df, pd.DataFrame)
    left = pd.DataFrame({'left_val': ['a', 'b', 'c']}, index=[1, 5, 10])
    right = pd.DataFrame({'right_val': [1, 2, 3, 6, 7]}, index=[1, 2, 3, 6, 7])
    with warns_that_defaulting_to_pandas():
        df = pd.merge_asof(left, right, left_index=True, right_index=True)
        assert isinstance(df, pd.DataFrame)
    with pytest.raises(ValueError):
        pd.merge_asof({'left_val': ['a', 'b', 'c']}, {'right_val': [1, 2, 3, 6, 7]}, left_index=True, right_index=True)