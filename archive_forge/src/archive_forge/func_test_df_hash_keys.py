import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_df_hash_keys():
    obj = DataFrame({'x': np.arange(3), 'y': list('abc')})
    a = hash_pandas_object(obj, hash_key='9876543210123456')
    b = hash_pandas_object(obj, hash_key='9876543210123465')
    assert (a != b).all()