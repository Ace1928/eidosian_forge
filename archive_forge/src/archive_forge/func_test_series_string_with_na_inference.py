from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.parametrize('na_value', [None, np.nan, pd.NA])
def test_series_string_with_na_inference(self, na_value):
    pytest.importorskip('pyarrow')
    dtype = 'string[pyarrow_numpy]'
    expected = Series(['a', na_value], dtype=dtype)
    with pd.option_context('future.infer_string', True):
        ser = Series(['a', na_value])
    tm.assert_series_equal(ser, expected)