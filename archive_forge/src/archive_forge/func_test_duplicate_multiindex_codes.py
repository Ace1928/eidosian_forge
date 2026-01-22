from itertools import product
import numpy as np
import pytest
from pandas._libs import (
from pandas import (
import pandas._testing as tm
def test_duplicate_multiindex_codes():
    msg = "Level values must be unique: \\[[A', ]+\\] on level 0"
    with pytest.raises(ValueError, match=msg):
        mi = MultiIndex([['A'] * 10, range(10)], [[0] * 10, range(10)])
    mi = MultiIndex.from_arrays([['A', 'A', 'B', 'B', 'B'], [1, 2, 1, 2, 3]])
    msg = "Level values must be unique: \\[[AB', ]+\\] on level 0"
    with pytest.raises(ValueError, match=msg):
        mi.set_levels([['A', 'B', 'A', 'A', 'B'], [2, 1, 3, -2, 5]])