from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_custom_business_day_freq(self):
    from pandas.tseries.offsets import CustomBusinessDay
    s = Series(range(100, 121), index=pd.bdate_range(start='2014-05-01', end='2014-06-01', freq=CustomBusinessDay(holidays=['2014-05-26'])))
    _check_plot_works(s.plot)