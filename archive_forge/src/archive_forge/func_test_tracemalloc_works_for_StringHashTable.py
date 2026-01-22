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
def test_tracemalloc_works_for_StringHashTable():
    N = 1000
    keys = np.arange(N).astype(np.str_).astype(np.object_)
    with activated_tracemalloc():
        table = ht.StringHashTable()
        table.map_locations(keys)
        used = get_allocated_khash_memory()
        my_size = table.sizeof()
        assert used == my_size
        del table
        assert get_allocated_khash_memory() == 0