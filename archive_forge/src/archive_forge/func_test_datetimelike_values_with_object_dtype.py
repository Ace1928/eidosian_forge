import array
from collections import (
from collections.abc import Iterator
from dataclasses import make_dataclass
from datetime import (
import functools
import re
import numpy as np
from numpy import ma
from numpy.ma import mrecords
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import lib
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.arrays import (
@pytest.mark.parametrize('kind', ['m', 'M'])
def test_datetimelike_values_with_object_dtype(self, kind, frame_or_series):
    if kind == 'M':
        dtype = 'M8[ns]'
        scalar_type = Timestamp
    else:
        dtype = 'm8[ns]'
        scalar_type = Timedelta
    arr = np.arange(6, dtype='i8').view(dtype).reshape(3, 2)
    if frame_or_series is Series:
        arr = arr[:, 0]
    obj = frame_or_series(arr, dtype=object)
    assert obj._mgr.arrays[0].dtype == object
    assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)
    obj = frame_or_series(frame_or_series(arr), dtype=object)
    assert obj._mgr.arrays[0].dtype == object
    assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)
    obj = frame_or_series(frame_or_series(arr), dtype=NumpyEADtype(object))
    assert obj._mgr.arrays[0].dtype == object
    assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)
    if frame_or_series is DataFrame:
        sers = [Series(x) for x in arr]
        obj = frame_or_series(sers, dtype=object)
        assert obj._mgr.arrays[0].dtype == object
        assert isinstance(obj._mgr.arrays[0].ravel()[0], scalar_type)