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
@pytest.mark.xfail(reason='Not clear what the correct expected behavior should be with integers now that we support non-nano. ATM (2022-10-08) we treat ints as nanoseconds, then cast to the requested dtype. xref #48312')
def test_constructor_dtype_timedelta_ns_s(self):
    result = Series([1000000, 200000, 3000000], dtype='timedelta64[ns]')
    expected = Series([1000000, 200000, 3000000], dtype='timedelta64[s]')
    tm.assert_series_equal(result, expected)