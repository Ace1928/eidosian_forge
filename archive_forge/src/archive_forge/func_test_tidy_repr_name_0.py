from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('arg', [100, 1001])
def test_tidy_repr_name_0(self, arg):
    ser = Series(np.random.randn(arg), name=0)
    rep_str = repr(ser)
    assert 'Name: 0' in rep_str