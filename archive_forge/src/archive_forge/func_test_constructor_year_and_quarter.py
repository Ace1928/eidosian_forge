import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_year_and_quarter(self):
    year = Series([2001, 2002, 2003])
    quarter = year - 2000
    msg = 'Constructing PeriodIndex from fields is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        idx = PeriodIndex(year=year, quarter=quarter)
    strs = [f'{t[0]:d}Q{t[1]:d}' for t in zip(quarter, year)]
    lops = list(map(Period, strs))
    p = PeriodIndex(lops)
    tm.assert_index_equal(p, idx)