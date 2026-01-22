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
def test_get_labels_groupby_for_Int64(writable):
    table = ht.Int64HashTable()
    vals = np.array([1, 2, -1, 2, 1, -1], dtype=np.int64)
    vals.flags.writeable = writable
    arr, unique = table.get_labels_groupby(vals)
    expected_arr = np.array([0, 1, -1, 1, 0, -1], dtype=np.intp)
    expected_unique = np.array([1, 2], dtype=np.int64)
    tm.assert_numpy_array_equal(arr, expected_arr)
    tm.assert_numpy_array_equal(unique, expected_unique)