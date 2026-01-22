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
@pytest.mark.parametrize('hashtable', [ht.PyObjectHashTable, ht.StringHashTable, ht.Float64HashTable, ht.Int64HashTable, ht.Int32HashTable, ht.UInt64HashTable])
def test_hashtable_large_sizehint(self, hashtable):
    size_hint = np.iinfo(np.uint32).max + 1
    hashtable(size_hint=size_hint)