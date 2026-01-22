from datetime import (
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.frequencies import to_offset
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_take_dont_lose_meta(self, tzstr):
    rng = date_range('1/1/2000', periods=20, tz=tzstr)
    result = rng.take(range(5))
    assert result.tz == rng.tz
    assert result.freq == rng.freq