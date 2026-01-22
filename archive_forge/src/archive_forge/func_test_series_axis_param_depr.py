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
def test_series_axis_param_depr(_test_series):
    warning_msg = "The 'axis' keyword in Series.resample is deprecated and will be removed in a future version."
    with tm.assert_produces_warning(FutureWarning, match=warning_msg):
        _test_series.resample('h', axis=0)