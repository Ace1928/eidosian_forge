from datetime import (
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_index_repr_in_frame_with_nan(self):
    i = Index([1, np.nan])
    s = Series([1, 2], index=i)
    exp = '1.0    1\nNaN    2\ndtype: int64'
    assert repr(s) == exp