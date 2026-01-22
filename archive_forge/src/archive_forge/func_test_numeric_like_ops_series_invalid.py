import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_numeric_like_ops_series_invalid(self):
    s = Series(Categorical([1, 2, 3, 4]))
    msg = 'Object with dtype category cannot perform the numpy op log'
    with pytest.raises(TypeError, match=msg):
        np.log(s)