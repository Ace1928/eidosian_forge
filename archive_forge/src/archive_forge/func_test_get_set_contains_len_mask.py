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
def test_get_set_contains_len_mask(self, table_type, dtype):
    if table_type == ht.PyObjectHashTable:
        pytest.skip('Mask not supported for object')
    index = 5
    table = table_type(55, uses_mask=True)
    assert len(table) == 0
    assert index not in table
    table.set_item(index, 42)
    assert len(table) == 1
    assert index in table
    assert table.get_item(index) == 42
    with pytest.raises(KeyError, match='NA'):
        table.get_na()
    table.set_item(index + 1, 41)
    table.set_na(41)
    assert pd.NA in table
    assert index in table
    assert index + 1 in table
    assert len(table) == 3
    assert table.get_item(index) == 42
    assert table.get_item(index + 1) == 41
    assert table.get_na() == 41
    table.set_na(21)
    assert index in table
    assert index + 1 in table
    assert len(table) == 3
    assert table.get_item(index + 1) == 41
    assert table.get_na() == 21
    assert index + 2 not in table
    with pytest.raises(KeyError, match=str(index + 2)):
        table.get_item(index + 2)