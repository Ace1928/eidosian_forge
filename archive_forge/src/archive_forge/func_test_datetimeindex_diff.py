from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_datetimeindex_diff(self, sort):
    dti1 = date_range(freq='QE-JAN', start=datetime(1997, 12, 31), periods=100)
    dti2 = date_range(freq='QE-JAN', start=datetime(1997, 12, 31), periods=98)
    assert len(dti1.difference(dti2, sort)) == 2