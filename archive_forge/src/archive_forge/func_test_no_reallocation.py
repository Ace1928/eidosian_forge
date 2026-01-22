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
@pytest.mark.parametrize('N', range(1, 110))
def test_no_reallocation(self, table_type, dtype, N):
    keys = np.arange(N).astype(dtype)
    preallocated_table = table_type(N)
    n_buckets_start = preallocated_table.get_state()['n_buckets']
    preallocated_table.map_locations(keys)
    n_buckets_end = preallocated_table.get_state()['n_buckets']
    assert n_buckets_start == n_buckets_end
    clean_table = table_type()
    clean_table.map_locations(keys)
    assert n_buckets_start == clean_table.get_state()['n_buckets']