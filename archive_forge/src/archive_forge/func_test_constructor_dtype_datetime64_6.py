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
def test_constructor_dtype_datetime64_6(self):
    ser = Series([None, NaT, '2013-08-05 15:30:00.000001'])
    assert ser.dtype == object
    ser = Series([np.nan, NaT, '2013-08-05 15:30:00.000001'])
    assert ser.dtype == object
    ser = Series([NaT, None, '2013-08-05 15:30:00.000001'])
    assert ser.dtype == object
    ser = Series([NaT, np.nan, '2013-08-05 15:30:00.000001'])
    assert ser.dtype == object