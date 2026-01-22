from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tzstr', ['US/Eastern', 'dateutil/US/Eastern'])
def test_dti_tz_nat(self, tzstr):
    idx = DatetimeIndex([Timestamp('2013-1-1', tz=tzstr), pd.NaT])
    assert isna(idx[1])
    assert idx[0].tzinfo is not None