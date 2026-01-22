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
@pytest.mark.parametrize('ts_value', [Timestamp('2000-01-01'), pd.NaT])
def test_frame_mixed_numeric_object_with_timestamp(ts_value):
    df = DataFrame({'a': [1], 'b': [1.1], 'c': ['foo'], 'd': [ts_value]})
    with pytest.raises(TypeError, match='does not support reduction'):
        df.sum()