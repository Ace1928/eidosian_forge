import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_format_sparse_display():
    index = MultiIndex(levels=[[0, 1], [0, 1], [0, 1], [0]], codes=[[0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]])
    msg = 'MultiIndex.format is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = index.format()
    assert result[3] == '1  0  0  0'