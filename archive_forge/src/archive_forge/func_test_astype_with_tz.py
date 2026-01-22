from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_with_tz(self):
    rng = date_range('1/1/2000', periods=10, tz='US/Eastern')
    msg = 'Cannot use .astype to convert from timezone-aware'
    with pytest.raises(TypeError, match=msg):
        rng.astype('datetime64[ns]')
    with pytest.raises(TypeError, match=msg):
        rng._data.astype('datetime64[ns]')