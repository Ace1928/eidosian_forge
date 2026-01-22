from datetime import timedelta
from decimal import Decimal
import re
from dateutil.tz import tzlocal
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import (
from pandas.compat.numpy import np_version_gt2
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import (
@pytest.fixture
def bool_frame_with_na():
    """
    Fixture for DataFrame of booleans with index of unique strings

    Columns are ['A', 'B', 'C', 'D']; some entries are missing
    """
    df = DataFrame(np.concatenate([np.ones((15, 4), dtype=bool), np.zeros((15, 4), dtype=bool)], axis=0), index=Index([f'foo_{i}' for i in range(30)], dtype=object), columns=Index(list('ABCD'), dtype=object), dtype=object)
    df.iloc[5:10] = np.nan
    df.iloc[15:20, -2:] = np.nan
    return df