import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_corr_invalid_method(self):
    df = DataFrame(np.random.default_rng(2).normal(size=(10, 2)))
    msg = "method must be either 'pearson', 'spearman', 'kendall', or a callable, "
    with pytest.raises(ValueError, match=msg):
        df.corr(method='____')