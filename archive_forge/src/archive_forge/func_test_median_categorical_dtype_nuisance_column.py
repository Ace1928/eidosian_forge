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
def test_median_categorical_dtype_nuisance_column(self):
    df = DataFrame({'A': Categorical([1, 2, 2, 2, 3])})
    ser = df['A']
    with pytest.raises(TypeError, match='does not support reduction'):
        ser.median()
    with pytest.raises(TypeError, match='does not support reduction'):
        df.median(numeric_only=False)
    with pytest.raises(TypeError, match='does not support reduction'):
        df.median()
    df['B'] = df['A'].astype(int)
    with pytest.raises(TypeError, match='does not support reduction'):
        df.median(numeric_only=False)
    with pytest.raises(TypeError, match='does not support reduction'):
        df.median()