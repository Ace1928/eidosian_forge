from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_str_freq_and_tz(self):
    dti = date_range('3/6/2012 00:00', periods=2, freq='h', tz='Europe/London', name='test_name')
    result = dti.astype(str)
    expected = Index(['2012-03-06 00:00:00+00:00', '2012-03-06 01:00:00+00:00'], dtype=object, name='test_name')
    tm.assert_index_equal(result, expected)