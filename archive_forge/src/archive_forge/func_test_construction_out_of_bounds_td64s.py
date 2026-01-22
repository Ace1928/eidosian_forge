from datetime import timedelta
from itertools import product
import numpy as np
import pytest
from pandas._libs.tslibs import OutOfBoundsTimedelta
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('val, unit', [(15251 * 10 ** 9, 'W'), (106752 * 10 ** 9, 'D'), (2562048 * 10 ** 9, 'h'), (153722868 * 10 ** 9, 'm')])
def test_construction_out_of_bounds_td64s(val, unit):
    td64 = np.timedelta64(val, unit)
    with pytest.raises(OutOfBoundsTimedelta, match=str(td64)):
        Timedelta(td64)
    assert Timedelta(td64 - 10 ** 9) == td64 - 10 ** 9