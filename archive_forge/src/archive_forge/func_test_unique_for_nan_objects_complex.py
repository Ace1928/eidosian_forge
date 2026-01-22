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
def test_unique_for_nan_objects_complex():
    table = ht.PyObjectHashTable()
    keys = np.array([complex(float('nan'), 1.0) for i in range(50)], dtype=np.object_)
    unique = table.unique(keys)
    assert len(unique) == 1