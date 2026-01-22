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
@pytest.mark.filterwarnings('ignore:The previous implementation of stack is deprecated')
def test_stack_multiple_out_of_bounds(self, multiindex_year_month_day_dataframe_random_data, future_stack):
    ymd = multiindex_year_month_day_dataframe_random_data
    unstacked = ymd.unstack(['year', 'month'])
    with pytest.raises(IndexError, match='Too many levels'):
        unstacked.stack([2, 3], future_stack=future_stack)
    with pytest.raises(IndexError, match='not a valid level number'):
        unstacked.stack([-4, -3], future_stack=future_stack)