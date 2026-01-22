import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_truncate_nonsortedindex_axis1(self):
    df = DataFrame({3: np.random.default_rng(2).standard_normal(5), 20: np.random.default_rng(2).standard_normal(5), 2: np.random.default_rng(2).standard_normal(5), 0: np.random.default_rng(2).standard_normal(5)}, columns=[3, 20, 2, 0])
    msg = 'truncate requires a sorted index'
    with pytest.raises(ValueError, match=msg):
        df.truncate(before=2, after=20, axis=1)