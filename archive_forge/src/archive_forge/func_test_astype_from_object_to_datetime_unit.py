import re
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('unit', ['Y', 'M', 'W', 'D', 'h', 'm'])
def test_astype_from_object_to_datetime_unit(self, unit):
    vals = [['2015-01-01', '2015-01-02', '2015-01-03'], ['2017-01-01', '2017-01-02', '2017-02-03']]
    df = DataFrame(vals, dtype=object)
    msg = f"Unexpected value for 'dtype': 'datetime64\\[{unit}\\]'. Must be 'datetime64\\[s\\]', 'datetime64\\[ms\\]', 'datetime64\\[us\\]', 'datetime64\\[ns\\]' or DatetimeTZDtype"
    with pytest.raises(ValueError, match=msg):
        df.astype(f'M8[{unit}]')