from datetime import (
import subprocess
import sys
import numpy as np
import pytest
import pandas._config.config as cf
from pandas._libs.tslibs import to_offset
from pandas import (
import pandas._testing as tm
from pandas.plotting import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('freq', ('B', 'ms', 's'))
def test_dateindex_conversion(self, freq, dtc):
    rtol = 10 ** (-9)
    dateindex = date_range('2020-01-01', periods=10, freq=freq)
    rs = dtc.convert(dateindex, None, None)
    xp = converter.mdates.date2num(dateindex._mpl_repr())
    tm.assert_almost_equal(rs, xp, rtol=rtol)