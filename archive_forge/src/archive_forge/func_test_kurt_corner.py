import inspect
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_kurt_corner(self):
    min_N = 4
    for i in range(1, min_N + 1):
        s = Series(np.ones(i))
        df = DataFrame(np.ones((i, i)))
        if i < min_N:
            assert np.isnan(s.kurt())
            assert np.isnan(df.kurt()).all()
        else:
            assert 0 == s.kurt()
            assert isinstance(s.kurt(), np.float64)
            assert (df.kurt() == 0).all()