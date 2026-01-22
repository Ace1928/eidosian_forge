import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_numeric_like_ops_series(self):
    s = Series(Categorical([1, 2, 3, 4]))
    with pytest.raises(TypeError, match="does not support reduction 'sum'"):
        np.sum(s)