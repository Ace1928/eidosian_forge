from datetime import (
import numpy as np
import pytest
from pandas._libs.algos import (
from pandas import (
import pandas._testing as tm
@pytest.mark.single_cpu
def test_pct_max_many_rows(self):
    df = DataFrame({'A': np.arange(2 ** 24 + 1), 'B': np.arange(2 ** 24 + 1, 0, -1)})
    result = df.rank(pct=True).max()
    assert (result == 1).all()