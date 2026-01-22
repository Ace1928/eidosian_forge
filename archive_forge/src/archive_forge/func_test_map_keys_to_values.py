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
def test_map_keys_to_values(self, table_type, dtype, writable):
    if table_type == ht.Int64HashTable:
        N = 77
        table = table_type()
        keys = np.arange(N).astype(dtype)
        vals = np.arange(N).astype(np.int64) + N
        keys.flags.writeable = writable
        vals.flags.writeable = writable
        table.map_keys_to_values(keys, vals)
        for i in range(N):
            assert table.get_item(keys[i]) == i + N