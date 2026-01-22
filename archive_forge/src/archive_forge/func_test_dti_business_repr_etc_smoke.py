from datetime import datetime
import dateutil.tz
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tz', [None, pytz.utc, dateutil.tz.tzutc()])
@pytest.mark.parametrize('freq', ['B', 'C'])
def test_dti_business_repr_etc_smoke(self, tz, freq):
    dti = pd.bdate_range(datetime(2009, 1, 1), datetime(2010, 1, 1), tz=tz, freq=freq)
    repr(dti)
    dti._summary()
    dti[2:2]._summary()