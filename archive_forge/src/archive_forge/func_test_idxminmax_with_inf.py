from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_idxminmax_with_inf(self):
    s = Series([0, -np.inf, np.inf, np.nan])
    assert s.idxmin() == 1
    msg = 'The behavior of Series.idxmin with all-NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert np.isnan(s.idxmin(skipna=False))
    assert s.idxmax() == 2
    msg = 'The behavior of Series.idxmax with all-NA values'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        assert np.isnan(s.idxmax(skipna=False))
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with pd.option_context('mode.use_inf_as_na', True):
            assert s.idxmin() == 0
            assert np.isnan(s.idxmin(skipna=False))
            assert s.idxmax() == 0
            np.isnan(s.idxmax(skipna=False))