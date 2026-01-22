from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('item', list({'days': 'D', 'seconds': 's', 'microseconds': 'us', 'milliseconds': 'ms', 'minutes': 'm', 'hours': 'h', 'weeks': 'W'}.items()))
@pytest.mark.parametrize('npdtype', [np.int64, np.int32, np.int16, np.float64, np.float32, np.float16])
def test_td_construction_with_np_dtypes(npdtype, item):
    pykwarg, npkwarg = item
    expected = np.timedelta64(1, npkwarg).astype('m8[ns]').view('i8')
    assert Timedelta(**{pykwarg: npdtype(1)})._value == expected