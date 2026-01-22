from collections.abc import Generator
from contextlib import contextmanager
import re
import struct
import tracemalloc
import numpy as np
import pytest
from pandas._libs import hashtable as ht
import pandas as pd
import pandas._testing as tm
from pandas.core.algorithms import isin
def test_value_count(self, dtype):
    values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
    keys, counts, _ = ht.value_count(values, True)
    assert len(keys) == 0
    keys, counts, _ = ht.value_count(values, False)
    assert len(keys) == 1 and np.all(np.isnan(keys))
    assert counts[0] == 3