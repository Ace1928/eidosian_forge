from datetime import (
import numpy as np
import pytest
import pytz
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tseries.offsets import (
def test_intersection_bug_1708(self):
    from pandas import DateOffset
    index_1 = date_range('1/1/2012', periods=4, freq='12h')
    index_2 = index_1 + DateOffset(hours=1)
    result = index_1.intersection(index_2)
    assert len(result) == 0