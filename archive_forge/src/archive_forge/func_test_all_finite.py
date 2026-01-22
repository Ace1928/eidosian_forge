from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_all_finite(self):
    alpha, beta = (0.3, 0.1)
    left_tailed = self.prng.beta(alpha, beta, size=100)
    assert nanops.nankurt(left_tailed) < 2
    alpha, beta = (0.1, 0.3)
    right_tailed = self.prng.beta(alpha, beta, size=100)
    assert nanops.nankurt(right_tailed) < 0