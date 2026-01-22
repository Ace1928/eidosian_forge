from datetime import datetime
import dateutil
import numpy as np
import pytest
import pytz
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_astype2(self):
    rng = date_range('1/1/2000', periods=10, name='idx')
    result = rng.astype('i8')
    tm.assert_index_equal(result, Index(rng.asi8, name='idx'))
    tm.assert_numpy_array_equal(result.values, rng.asi8)