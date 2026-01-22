import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's', 'h', 'm', 'D'])
def test_astype_to_incorrect_datetimelike(self, unit):
    dtype = f'M8[{unit}]'
    other = f'm8[{unit}]'
    df = DataFrame(np.array([[1, 2, 3]], dtype=dtype))
    msg = '|'.join([f'Cannot cast DatetimeArray to dtype timedelta64\\[{unit}\\]', f'cannot astype a datetimelike from \\[datetime64\\[ns\\]\\] to \\[timedelta64\\[{unit}\\]\\]'])
    with pytest.raises(TypeError, match=msg):
        df.astype(other)
    msg = '|'.join([f'Cannot cast TimedeltaArray to dtype datetime64\\[{unit}\\]', f'cannot astype a timedelta from \\[timedelta64\\[ns\\]\\] to \\[datetime64\\[{unit}\\]\\]'])
    df = DataFrame(np.array([[1, 2, 3]], dtype=other))
    with pytest.raises(TypeError, match=msg):
        df.astype(dtype)