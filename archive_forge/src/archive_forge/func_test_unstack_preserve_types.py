from datetime import datetime
import itertools
import re
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.reshape import reshape as reshape_lib
def test_unstack_preserve_types(self, multiindex_year_month_day_dataframe_random_data, using_infer_string):
    ymd = multiindex_year_month_day_dataframe_random_data
    ymd['E'] = 'foo'
    ymd['F'] = 2
    unstacked = ymd.unstack('month')
    assert unstacked['A', 1].dtype == np.float64
    assert unstacked['E', 1].dtype == np.object_ if not using_infer_string else 'string'
    assert unstacked['F', 1].dtype == np.float64