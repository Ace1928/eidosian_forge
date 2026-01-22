from datetime import (
import numpy as np
import pytest
from pandas.compat import (
from pandas import (
import pandas._testing as tm
from pandas.api.indexers import BaseIndexer
from pandas.core.indexers.objects import VariableOffsetWindowIndexer
from pandas.tseries.offsets import BusinessDay
@pytest.mark.parametrize('window', [timedelta(days=3), Timedelta(days=3), '3D', VariableOffsetWindowIndexer(index=date_range('2015-12-25', periods=5), offset=BusinessDay(1))])
def test_freq_window_not_implemented(window):
    df = DataFrame(np.arange(10), index=date_range('2015-12-24', periods=10, freq='D'))
    with pytest.raises(NotImplementedError, match='^step (not implemented|is not supported)'):
        df.rolling(window, step=3).sum()