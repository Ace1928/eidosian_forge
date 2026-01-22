import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dup_columns_across_dtype(self):
    vals = [[1, -1, 2.0], [2, -2, 3.0]]
    rs = DataFrame(vals, columns=['A', 'A', 'B'])
    xp = DataFrame(vals)
    xp.columns = ['A', 'A', 'B']
    tm.assert_frame_equal(rs, xp)