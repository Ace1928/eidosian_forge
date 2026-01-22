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
def test_lookup_nan(self, writable):
    xs = np.array([2.718, 3.14, np.nan, -7, 5, 2, 3])
    xs.setflags(write=writable)
    m = ht.Float64HashTable()
    m.map_locations(xs)
    tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))