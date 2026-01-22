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
@pytest.mark.parametrize('index', [date_range('1/1/2000', periods=10), timedelta_range('1 day', periods=10), period_range('2000-Q1', periods=10, freq='Q')], ids=lambda x: type(x).__name__)
def test_constructor_cant_cast_datetimelike(self, index):
    msg = f'Cannot cast {type(index).__name__.rstrip('Index')}.*? to '
    with pytest.raises(TypeError, match=msg):
        Series(index, dtype=float)
    result = Series(index, dtype=np.int64)
    expected = Series(index.astype(np.int64))
    tm.assert_series_equal(result, expected)