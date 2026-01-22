import operator
import numpy as np
import pytest
from pandas.compat.pyarrow import pa_version_under12p0
from pandas.core.dtypes.common import is_dtype_equal
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays.string_arrow import (
def test_constructor_raises(cls):
    if cls is pd.arrays.StringArray:
        msg = 'StringArray requires a sequence of strings or pandas.NA'
    else:
        msg = "Unsupported type '<class 'numpy.ndarray'>' for ArrowExtensionArray"
    with pytest.raises(ValueError, match=msg):
        cls(np.array(['a', 'b'], dtype='S1'))
    with pytest.raises(ValueError, match=msg):
        cls(np.array([]))
    if cls is pd.arrays.StringArray:
        cls(np.array(['a', np.nan], dtype=object))
        cls(np.array(['a', None], dtype=object))
    else:
        with pytest.raises(ValueError, match=msg):
            cls(np.array(['a', np.nan], dtype=object))
        with pytest.raises(ValueError, match=msg):
            cls(np.array(['a', None], dtype=object))
    with pytest.raises(ValueError, match=msg):
        cls(np.array(['a', pd.NaT], dtype=object))
    with pytest.raises(ValueError, match=msg):
        cls(np.array(['a', np.datetime64('NaT', 'ns')], dtype=object))
    with pytest.raises(ValueError, match=msg):
        cls(np.array(['a', np.timedelta64('NaT', 'ns')], dtype=object))