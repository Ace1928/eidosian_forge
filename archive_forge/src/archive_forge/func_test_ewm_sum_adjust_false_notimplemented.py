import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewm_sum_adjust_false_notimplemented():
    data = Series(range(1)).ewm(com=1, adjust=False)
    with pytest.raises(NotImplementedError, match='sum is not'):
        data.sum()