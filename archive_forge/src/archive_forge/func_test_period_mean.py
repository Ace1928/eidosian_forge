import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('box', [Series, pd.Index, pd.array])
@pytest.mark.parametrize('freq', ['s', 'h', 'D', 'W', 'B'])
def test_period_mean(self, box, freq):
    dti = date_range('2001-01-01', periods=11)
    dti = dti.take([4, 1, 3, 10, 9, 7, 8, 5, 0, 2, 6])
    warn = FutureWarning if freq == 'B' else None
    msg = 'PeriodDtype\\[B\\] is deprecated'
    with tm.assert_produces_warning(warn, match=msg):
        parr = dti._data.to_period(freq)
    obj = box(parr)
    with pytest.raises(TypeError, match='ambiguous'):
        obj.mean()
    with pytest.raises(TypeError, match='ambiguous'):
        obj.mean(skipna=True)
    parr[-2] = pd.NaT
    with pytest.raises(TypeError, match='ambiguous'):
        obj.mean()
    with pytest.raises(TypeError, match='ambiguous'):
        obj.mean(skipna=True)