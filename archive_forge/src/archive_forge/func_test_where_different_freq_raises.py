import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
@pytest.mark.parametrize('other', [pd.Period('2000', freq='h'), PeriodArray._from_sequence(['2000', '2001', '2000'], dtype='period[h]')])
def test_where_different_freq_raises(other):
    ser = pd.Series(PeriodArray._from_sequence(['2000', '2001', '2002'], dtype='period[D]'))
    cond = np.array([True, False, True])
    with pytest.raises(IncompatibleFrequency, match='freq'):
        ser.array._where(cond, other)
    res = ser.where(cond, other)
    expected = ser.astype(object).where(cond, other)
    tm.assert_series_equal(res, expected)