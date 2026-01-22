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
def test_infer_with_date_and_datetime(self):
    ts = Timestamp(2016, 1, 1)
    vals = [ts.to_pydatetime(), ts.date()]
    ser = Series(vals)
    expected = Series(vals, dtype=object)
    tm.assert_series_equal(ser, expected)
    idx = Index(vals)
    expected = Index(vals, dtype=object)
    tm.assert_index_equal(idx, expected)