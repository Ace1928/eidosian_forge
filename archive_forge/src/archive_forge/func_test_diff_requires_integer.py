import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_diff_requires_integer(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((2, 2)))
    with pytest.raises(ValueError, match='periods must be an integer'):
        df.diff(1.5)