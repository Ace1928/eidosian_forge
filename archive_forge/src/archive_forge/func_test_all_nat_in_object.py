import operator
import numpy as np
import pytest
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_all_nat_in_object(self):
    now = pd.Timestamp.now('UTC')
    df = DataFrame({'a': pd.to_datetime([None, None], utc=True)}, dtype=object)
    result = df.query('a > @now')
    expected = DataFrame({'a': []}, dtype=object)
    tm.assert_frame_equal(result, expected)