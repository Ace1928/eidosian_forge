from itertools import permutations
import re
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
def test_is_unique_interval(self, closed):
    """
        Interval specific tests for is_unique in addition to base class tests
        """
    idx = IntervalIndex.from_tuples([(0, 1), (0.5, 1.5)], closed=closed)
    assert idx.is_unique is True
    idx = IntervalIndex.from_tuples([(1, 2), (1, 3), (2, 3)], closed=closed)
    assert idx.is_unique is True
    idx = IntervalIndex.from_tuples([(-1, 1), (-2, 2)], closed=closed)
    assert idx.is_unique is True
    idx = IntervalIndex.from_tuples([(np.nan, np.nan)], closed=closed)
    assert idx.is_unique is True
    idx = IntervalIndex.from_tuples([(np.nan, np.nan), (np.nan, np.nan)], closed=closed)
    assert idx.is_unique is False