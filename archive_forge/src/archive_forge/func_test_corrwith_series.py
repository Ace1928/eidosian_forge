import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corrwith_series(self, datetime_frame):
    result = datetime_frame.corrwith(datetime_frame['A'])
    expected = datetime_frame.apply(datetime_frame['A'].corr)
    tm.assert_series_equal(result, expected)