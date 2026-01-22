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
def test_nan_in_tuple(self):
    nan1 = (float('nan'),)
    nan2 = (float('nan'),)
    assert nan1[0] is not nan2[0]
    table = ht.PyObjectHashTable()
    table.set_item(nan1, 42)
    assert table.get_item(nan2) == 42