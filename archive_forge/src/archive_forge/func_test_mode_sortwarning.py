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
def test_mode_sortwarning(self, using_infer_string):
    df = DataFrame({'A': [np.nan, np.nan, 'a', 'a']})
    expected = DataFrame({'A': ['a', np.nan]})
    warning = None if using_infer_string else UserWarning
    with tm.assert_produces_warning(warning):
        result = df.mode(dropna=False)
        result = result.sort_values(by='A').reset_index(drop=True)
    tm.assert_frame_equal(result, expected)