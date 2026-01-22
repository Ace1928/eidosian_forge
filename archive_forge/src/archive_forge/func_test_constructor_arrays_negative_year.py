import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_constructor_arrays_negative_year(self):
    years = np.arange(1960, 2000, dtype=np.int64).repeat(4)
    quarters = np.tile(np.array([1, 2, 3, 4], dtype=np.int64), 40)
    msg = 'Constructing PeriodIndex from fields is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        pindex = PeriodIndex(year=years, quarter=quarters)
    tm.assert_index_equal(pindex.year, Index(years))
    tm.assert_index_equal(pindex.quarter, Index(quarters))
    alt = PeriodIndex.from_fields(year=years, quarter=quarters)
    tm.assert_index_equal(alt, pindex)