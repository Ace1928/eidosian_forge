import numpy as np
import pytest
from pandas.compat.numpy import np_version_gte1p25
import pandas as pd
import pandas._testing as tm
def test_reconstruct_func():
    result = pd.core.apply.reconstruct_func('min')
    expected = (False, 'min', None, None)
    tm.assert_equal(result, expected)