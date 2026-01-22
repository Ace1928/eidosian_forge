import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hash_tuples():
    tuples = [(1, 'one'), (1, 'two'), (2, 'one')]
    result = hash_tuples(tuples)
    expected = hash_pandas_object(MultiIndex.from_tuples(tuples)).values
    tm.assert_numpy_array_equal(result, expected)
    msg = '|'.join(['object is not iterable', 'zip argument #1 must support iteration'])
    with pytest.raises(TypeError, match=msg):
        hash_tuples(tuples[0])