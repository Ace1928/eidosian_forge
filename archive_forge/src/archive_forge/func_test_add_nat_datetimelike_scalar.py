from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import Timedelta
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('nat', [np.datetime64('NaT', 'ns'), np.datetime64('NaT', 'us')])
def test_add_nat_datetimelike_scalar(self, nat, tda):
    result = tda + nat
    assert isinstance(result, DatetimeArray)
    assert result._creso == tda._creso
    assert result.isna().all()
    result = nat + tda
    assert isinstance(result, DatetimeArray)
    assert result._creso == tda._creso
    assert result.isna().all()