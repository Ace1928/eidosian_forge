from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('operation', [nanops.nanany, nanops.nanall, nanops.nansum, nanops.nanmean, nanops.nanmedian, nanops.nanstd, nanops.nanvar, nanops.nansem, nanops.nanargmax, nanops.nanargmin, nanops.nanmax, nanops.nanmin, nanops.nanskew, nanops.nankurt, nanops.nanprod])
def test_nanops_independent_of_mask_param(operation):
    ser = Series([1, 2, np.nan, 3, np.nan, 4])
    mask = ser.isna()
    median_expected = operation(ser._values)
    median_result = operation(ser._values, mask=mask)
    assert median_expected == median_result