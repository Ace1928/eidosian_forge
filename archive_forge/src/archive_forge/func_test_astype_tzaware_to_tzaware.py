from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype_tzaware_to_tzaware(self):
    idx = date_range('20170101', periods=4, tz='US/Pacific')
    result = idx.astype('datetime64[ns, US/Eastern]')
    expected = date_range('20170101 03:00:00', periods=4, tz='US/Eastern')
    tm.assert_index_equal(result, expected)
    assert result.freq == expected.freq