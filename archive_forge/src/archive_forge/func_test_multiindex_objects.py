import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_multiindex_objects():
    mi = MultiIndex(levels=[['b', 'd', 'a'], [1, 2, 3]], codes=[[0, 1, 0, 2], [2, 0, 0, 1]], names=['col1', 'col2'])
    recons = mi._sort_levels_monotonic()
    assert mi.equals(recons)
    assert Index(mi.values).equals(Index(recons.values))