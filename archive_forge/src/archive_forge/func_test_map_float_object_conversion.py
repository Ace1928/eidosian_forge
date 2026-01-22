from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import BDay
@pytest.mark.parametrize('val', [1, 1.0])
def test_map_float_object_conversion(val):
    df = DataFrame(data=[val, 'a'])
    result = df.map(lambda x: x).dtypes[0]
    assert result == object