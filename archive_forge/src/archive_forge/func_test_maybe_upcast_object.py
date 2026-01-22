import numpy as np
import pytest
from pandas._libs.parsers import (
import pandas as pd
from pandas import NA
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('val', [na_values[np.object_], 'c'])
def test_maybe_upcast_object(val, string_storage):
    pa = pytest.importorskip('pyarrow')
    with pd.option_context('mode.string_storage', string_storage):
        arr = np.array(['a', 'b', val], dtype=np.object_)
        result = _maybe_upcast(arr, use_dtype_backend=True)
        if string_storage == 'python':
            exp_val = 'c' if val == 'c' else NA
            expected = StringArray(np.array(['a', 'b', exp_val], dtype=np.object_))
        else:
            exp_val = 'c' if val == 'c' else None
            expected = ArrowStringArray(pa.array(['a', 'b', exp_val]))
        tm.assert_extension_array_equal(result, expected)