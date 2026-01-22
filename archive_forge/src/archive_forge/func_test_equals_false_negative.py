from contextlib import nullcontext
import copy
import numpy as np
import pytest
from pandas._libs.missing import is_matching_na
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import is_float
from pandas import (
import pandas._testing as tm
def test_equals_false_negative():
    arr = [False, np.nan]
    s1 = Series(arr)
    s2 = s1.copy()
    s3 = Series(index=range(2), dtype=object)
    s4 = s3.copy()
    s5 = s3.copy()
    s6 = s3.copy()
    s3[:-1] = s4[:-1] = s5[0] = s6[0] = False
    assert s1.equals(s1)
    assert s1.equals(s2)
    assert s1.equals(s3)
    assert s1.equals(s4)
    assert s1.equals(s5)
    assert s5.equals(s6)