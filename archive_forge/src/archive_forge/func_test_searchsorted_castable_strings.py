from __future__ import annotations
import re
import warnings
import numpy as np
import pytest
from pandas._libs import (
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('box', [None, 'index', 'series'])
def test_searchsorted_castable_strings(self, arr1d, box, string_storage):
    arr = arr1d
    if box is None:
        pass
    elif box == 'index':
        arr = self.index_cls(arr)
    else:
        arr = pd.Series(arr)
    result = arr.searchsorted(str(arr[1]))
    assert result == 1
    result = arr.searchsorted(str(arr[2]), side='right')
    assert result == 3
    result = arr.searchsorted([str(x) for x in arr[1:3]])
    expected = np.array([1, 2], dtype=np.intp)
    tm.assert_numpy_array_equal(result, expected)
    with pytest.raises(TypeError, match=re.escape(f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got 'str' instead.")):
        arr.searchsorted('foo')
    with pd.option_context('string_storage', string_storage):
        with pytest.raises(TypeError, match=re.escape(f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', or array of those. Got string array instead.")):
            arr.searchsorted([str(arr[1]), 'baz'])