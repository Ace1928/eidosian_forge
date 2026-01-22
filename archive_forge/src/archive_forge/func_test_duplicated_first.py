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
def test_duplicated_first(self, dtype):
    values = np.array([np.nan, np.nan, np.nan], dtype=dtype)
    result = ht.duplicated(values)
    expected = np.array([False, True, True])
    tm.assert_numpy_array_equal(result, expected)