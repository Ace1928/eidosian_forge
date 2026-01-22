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
def test_merge_asof_bad_arguments():
    left = {'a': [1, 5, 10], 'b': [5, 7, 9]}
    right = {'a': [2, 3, 6], 'b': [6, 5, 20]}
    pandas_left, pandas_right = (pandas.DataFrame(left), pandas.DataFrame(right))
    modin_left, modin_right = (pd.DataFrame(left), pd.DataFrame(right))
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right, on='a', by='b', left_by="can't do with by")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on='a', by='b', left_by="can't do with by")
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right, by='b', on='a', right_by="can't do with by")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, by='b', on='a', right_by="can't do with by")
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right, on='a', left_on="can't do with by")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on='a', left_on="can't do with by")
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right, on='a', right_on="can't do with by")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on='a', right_on="can't do with by")
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on='a', right_index=True)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, left_on='a', right_on='a', right_index=True)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, on='a', left_index=True)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, left_on='a', right_on='a', left_index=True)
    with pytest.raises(Exception):
        pandas.merge_asof(pandas_left, pandas_right, left_on='a')
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, left_on='a')
    with pytest.raises(Exception):
        pandas.merge_asof(pandas_left, pandas_right, right_on='a')
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right, right_on='a')
    with pytest.raises(ValueError):
        pandas.merge_asof(pandas_left, pandas_right)
    with pytest.raises(ValueError), warns_that_defaulting_to_pandas():
        pd.merge_asof(modin_left, modin_right)