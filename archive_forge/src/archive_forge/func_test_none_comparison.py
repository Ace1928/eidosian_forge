from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_none_comparison(request, series_with_simple_index):
    series = series_with_simple_index
    if len(series) < 1:
        request.applymarker(pytest.mark.xfail(reason="Test doesn't make sense on empty data"))
    series.iloc[0] = np.nan
    result = series == None
    assert not result.iat[0]
    assert not result.iat[1]
    result = series != None
    assert result.iat[0]
    assert result.iat[1]
    result = None == series
    assert not result.iat[0]
    assert not result.iat[1]
    result = None != series
    assert result.iat[0]
    assert result.iat[1]
    if lib.is_np_dtype(series.dtype, 'M') or isinstance(series.dtype, DatetimeTZDtype):
        msg = 'Invalid comparison'
        with pytest.raises(TypeError, match=msg):
            None > series
        with pytest.raises(TypeError, match=msg):
            series > None
    else:
        result = None > series
        assert not result.iat[0]
        assert not result.iat[1]
        result = series < None
        assert not result.iat[0]
        assert not result.iat[1]