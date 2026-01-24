import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hash_keys():
    obj = Series(list('abc'))
    a = hash_pandas_object(obj, hash_key='9892843210123456')
    b = hash_pandas_object(obj, hash_key='9892843210123465')
    assert (a != b).all()