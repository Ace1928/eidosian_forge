from datetime import datetime
import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.fixture
def tdi(self, monotonic):
    tdi = timedelta_range('1 Day', periods=10)
    if monotonic == 'decreasing':
        tdi = tdi[::-1]
    elif monotonic is None:
        taker = np.arange(10, dtype=np.intp)
        np.random.default_rng(2).shuffle(taker)
        tdi = tdi.take(taker)
    return tdi