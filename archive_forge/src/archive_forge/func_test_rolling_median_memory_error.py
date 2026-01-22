from datetime import datetime
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def test_rolling_median_memory_error():
    n = 20000
    Series(np.random.default_rng(2).standard_normal(n)).rolling(window=2, center=False).median()
    Series(np.random.default_rng(2).standard_normal(n)).rolling(window=2, center=False).median()