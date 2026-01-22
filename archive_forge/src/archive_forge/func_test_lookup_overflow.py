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
def test_lookup_overflow(self, writable):
    xs = np.array([1, 2, 2 ** 63], dtype=np.uint64)
    xs.setflags(write=writable)
    m = ht.UInt64HashTable()
    m.map_locations(xs)
    tm.assert_numpy_array_equal(m.lookup(xs), np.arange(len(xs), dtype=np.intp))