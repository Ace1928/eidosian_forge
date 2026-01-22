from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_constructor_from_sparse_array(self):
    values = [Timestamp('2012-05-01T01:00:00.000000'), Timestamp('2016-05-01T01:00:00.000000')]
    arr = pd.arrays.SparseArray(values)
    result = Index(arr)
    assert type(result) is Index
    assert result.dtype == arr.dtype