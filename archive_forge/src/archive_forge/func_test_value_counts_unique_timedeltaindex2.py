import numpy as np
from pandas import (
import pandas._testing as tm
def test_value_counts_unique_timedeltaindex2(self):
    idx = TimedeltaIndex(['1 days 09:00:00', '1 days 09:00:00', '1 days 09:00:00', '1 days 08:00:00', '1 days 08:00:00', NaT])
    self._check_value_counts_dropna(idx)