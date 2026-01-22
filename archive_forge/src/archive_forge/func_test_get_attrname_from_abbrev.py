import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
import pandas._testing as tm
@pytest.mark.parametrize('freqstr,expected', [('Y', 'year'), ('Q', 'quarter'), ('M', 'month'), ('D', 'day'), ('h', 'hour'), ('min', 'minute'), ('s', 'second'), ('ms', 'millisecond'), ('us', 'microsecond'), ('ns', 'nanosecond')])
def test_get_attrname_from_abbrev(freqstr, expected):
    reso = Resolution.get_reso_from_freqstr(freqstr)
    assert reso.attr_abbrev == freqstr
    assert reso.attrname == expected