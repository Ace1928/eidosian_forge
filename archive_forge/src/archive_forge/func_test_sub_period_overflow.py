import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import PeriodDtype
import pandas as pd
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_sub_period_overflow():
    dti = pd.date_range('1677-09-22', periods=2, freq='D')
    pi = dti.to_period('ns')
    per = pd.Period._from_ordinal(10 ** 14, pi.freq)
    with pytest.raises(OverflowError, match='Overflow in int64 addition'):
        pi - per
    with pytest.raises(OverflowError, match='Overflow in int64 addition'):
        per - pi