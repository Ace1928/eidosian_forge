from datetime import datetime
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.datetimes import date_range
def test_try_aggregate_non_existing_column():
    data = [{'dt': datetime(2017, 6, 1, 0), 'x': 1.0, 'y': 2.0}, {'dt': datetime(2017, 6, 1, 1), 'x': 2.0, 'y': 2.0}, {'dt': datetime(2017, 6, 1, 2), 'x': 3.0, 'y': 1.5}]
    df = DataFrame(data).set_index('dt')
    msg = "Column\\(s\\) \\['z'\\] do not exist"
    with pytest.raises(KeyError, match=msg):
        df.resample('30min').agg({'x': ['mean'], 'y': ['median'], 'z': ['sum']})