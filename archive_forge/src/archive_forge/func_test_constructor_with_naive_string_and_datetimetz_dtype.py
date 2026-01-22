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
@pytest.mark.parametrize('arg', ['2013-01-01 00:00:00', NaT, np.nan, None])
def test_constructor_with_naive_string_and_datetimetz_dtype(self, arg):
    result = Series([arg], dtype='datetime64[ns, CET]')
    expected = Series(Timestamp(arg)).dt.tz_localize('CET')
    tm.assert_series_equal(result, expected)