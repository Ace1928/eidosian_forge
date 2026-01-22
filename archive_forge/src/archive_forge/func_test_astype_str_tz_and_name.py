from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_str_tz_and_name(self):
    dti = date_range('2012-01-01', periods=3, name='test_name', tz='US/Eastern')
    result = dti.astype(str)
    expected = Index(['2012-01-01 00:00:00-05:00', '2012-01-02 00:00:00-05:00', '2012-01-03 00:00:00-05:00'], name='test_name', dtype=object)
    tm.assert_index_equal(result, expected)