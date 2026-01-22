from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_union2(self, sort):
    everything = date_range('2020-01-01', periods=10)
    first = everything[:5]
    second = everything[5:]
    union = first.union(second, sort=sort)
    tm.assert_index_equal(union, everything)