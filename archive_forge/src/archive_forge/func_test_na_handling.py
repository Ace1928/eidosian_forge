import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('labels', [None, False])
def test_na_handling(labels):
    arr = np.arange(0, 0.75, 0.01)
    arr[::3] = np.nan
    result = cut(arr, 4, labels=labels)
    result = np.asarray(result)
    expected = np.where(isna(arr), np.nan, result)
    tm.assert_almost_equal(result, expected)