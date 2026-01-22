from datetime import (
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_equals_with_tz(self):
    left = date_range('1/1/2011', periods=100, freq='h', tz='utc')
    right = date_range('1/1/2011', periods=100, freq='h', tz='US/Eastern')
    assert not left.equals(right)