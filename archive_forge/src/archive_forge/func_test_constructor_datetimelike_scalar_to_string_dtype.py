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
def test_constructor_datetimelike_scalar_to_string_dtype(self, nullable_string_dtype):
    result = Series('M', index=[1, 2, 3], dtype=nullable_string_dtype)
    expected = Series(['M', 'M', 'M'], index=[1, 2, 3], dtype=nullable_string_dtype)
    tm.assert_series_equal(result, expected)