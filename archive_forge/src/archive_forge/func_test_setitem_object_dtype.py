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
@pytest.mark.parametrize('box', [pd.Index, pd.Series, np.array, list, NumpyExtensionArray])
def test_setitem_object_dtype(self, box, arr1d):
    expected = arr1d.copy()[::-1]
    if expected.dtype.kind in ['m', 'M']:
        expected = expected._with_freq(None)
    vals = expected
    if box is list:
        vals = list(vals)
    elif box is np.array:
        vals = np.array(vals.astype(object))
    elif box is NumpyExtensionArray:
        vals = box(np.asarray(vals, dtype=object))
    else:
        vals = box(vals).astype(object)
    arr1d[:] = vals
    tm.assert_equal(arr1d, expected)