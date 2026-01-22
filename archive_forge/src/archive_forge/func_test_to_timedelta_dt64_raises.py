from datetime import (
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import OutOfBoundsTimedelta
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import TimedeltaArray
def test_to_timedelta_dt64_raises(self):
    msg = 'dtype datetime64\\[ns\\] cannot be converted to timedelta64\\[ns\\]'
    ser = Series([pd.NaT])
    with pytest.raises(TypeError, match=msg):
        to_timedelta(ser)
    with pytest.raises(TypeError, match=msg):
        ser.to_frame().apply(to_timedelta)