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
def test_constructor_periodindex(self):
    pi = period_range('20130101', periods=5, freq='D')
    s = Series(pi)
    assert s.dtype == 'Period[D]'
    with tm.assert_produces_warning(FutureWarning, match='Dtype inference'):
        expected = Series(pi.astype(object))
    tm.assert_series_equal(s, expected)