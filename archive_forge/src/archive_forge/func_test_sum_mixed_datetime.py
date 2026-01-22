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
def test_sum_mixed_datetime(self):
    df = DataFrame({'A': date_range('2000', periods=4), 'B': [1, 2, 3, 4]}).reindex([2, 3, 4])
    with pytest.raises(TypeError, match="does not support reduction 'sum'"):
        df.sum()