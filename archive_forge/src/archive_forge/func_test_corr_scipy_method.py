import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('method', ['pearson', 'kendall', 'spearman'])
def test_corr_scipy_method(self, float_frame, method):
    pytest.importorskip('scipy')
    float_frame.loc[float_frame.index[:5], 'A'] = np.nan
    float_frame.loc[float_frame.index[5:10], 'B'] = np.nan
    float_frame.loc[float_frame.index[:10], 'A'] = float_frame['A'][10:20].copy()
    correls = float_frame.corr(method=method)
    expected = float_frame['A'].corr(float_frame['C'], method=method)
    tm.assert_almost_equal(correls['A']['C'], expected)