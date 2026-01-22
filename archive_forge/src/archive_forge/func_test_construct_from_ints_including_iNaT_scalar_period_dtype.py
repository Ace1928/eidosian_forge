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
@pytest.mark.xfail(reason='PeriodDtype Series not supported yet')
def test_construct_from_ints_including_iNaT_scalar_period_dtype(self):
    series = Series([0, 1000, 2000, pd._libs.iNaT], dtype='period[D]')
    val = series[3]
    assert isna(val)
    series[2] = val
    assert isna(series[2])