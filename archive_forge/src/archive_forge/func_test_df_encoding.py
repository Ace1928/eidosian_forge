import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_df_encoding():
    obj = DataFrame({'x': np.arange(3), 'y': list('a+c')})
    a = hash_pandas_object(obj, encoding='utf8')
    b = hash_pandas_object(obj, encoding='utf7')
    assert a[0] == b[0]
    assert a[1] != b[1]
    assert a[2] == b[2]