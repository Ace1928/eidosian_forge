from datetime import timedelta
import sys
from hypothesis import (
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.errors import OutOfBoundsTimedelta
from pandas import (
import pandas._testing as tm
def test_short_format_converters(self):

    def conv(v):
        return v.astype('m8[ns]')
    assert Timedelta('10') == np.timedelta64(10, 'ns')
    assert Timedelta('10ns') == np.timedelta64(10, 'ns')
    assert Timedelta('100') == np.timedelta64(100, 'ns')
    assert Timedelta('100ns') == np.timedelta64(100, 'ns')
    assert Timedelta('1000') == np.timedelta64(1000, 'ns')
    assert Timedelta('1000ns') == np.timedelta64(1000, 'ns')
    assert Timedelta('1000NS') == np.timedelta64(1000, 'ns')
    assert Timedelta('10us') == np.timedelta64(10000, 'ns')
    assert Timedelta('100us') == np.timedelta64(100000, 'ns')
    assert Timedelta('1000us') == np.timedelta64(1000000, 'ns')
    assert Timedelta('1000Us') == np.timedelta64(1000000, 'ns')
    assert Timedelta('1000uS') == np.timedelta64(1000000, 'ns')
    assert Timedelta('1ms') == np.timedelta64(1000000, 'ns')
    assert Timedelta('10ms') == np.timedelta64(10000000, 'ns')
    assert Timedelta('100ms') == np.timedelta64(100000000, 'ns')
    assert Timedelta('1000ms') == np.timedelta64(1000000000, 'ns')
    assert Timedelta('-1s') == -np.timedelta64(1000000000, 'ns')
    assert Timedelta('1s') == np.timedelta64(1000000000, 'ns')
    assert Timedelta('10s') == np.timedelta64(10000000000, 'ns')
    assert Timedelta('100s') == np.timedelta64(100000000000, 'ns')
    assert Timedelta('1000s') == np.timedelta64(1000000000000, 'ns')
    assert Timedelta('1d') == conv(np.timedelta64(1, 'D'))
    assert Timedelta('-1d') == -conv(np.timedelta64(1, 'D'))
    assert Timedelta('1D') == conv(np.timedelta64(1, 'D'))
    assert Timedelta('10D') == conv(np.timedelta64(10, 'D'))
    assert Timedelta('100D') == conv(np.timedelta64(100, 'D'))
    assert Timedelta('1000D') == conv(np.timedelta64(1000, 'D'))
    assert Timedelta('10000D') == conv(np.timedelta64(10000, 'D'))
    assert Timedelta(' 10000D ') == conv(np.timedelta64(10000, 'D'))
    assert Timedelta(' - 10000D ') == -conv(np.timedelta64(10000, 'D'))
    msg = 'invalid unit abbreviation'
    with pytest.raises(ValueError, match=msg):
        Timedelta('1foo')
    msg = 'unit abbreviation w/o a number'
    with pytest.raises(ValueError, match=msg):
        Timedelta('foo')