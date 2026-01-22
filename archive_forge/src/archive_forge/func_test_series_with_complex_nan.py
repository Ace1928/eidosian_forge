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
@pytest.mark.parametrize('input_list', [[1, complex('nan'), 2], [1 + 1j, complex('nan'), 2 + 2j]])
def test_series_with_complex_nan(input_list):
    ser = Series(input_list)
    result = Series(ser.array)
    assert ser.dtype == 'complex128'
    tm.assert_series_equal(ser, result)