from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
@pytest.mark.parametrize('htable, data', [(ht.PyObjectHashTable, [f'foo_{i}' for i in range(1000)]), (ht.StringHashTable, [f'foo_{i}' for i in range(1000)]), (ht.Float64HashTable, np.arange(1000, dtype=np.float64)), (ht.Int64HashTable, np.arange(1000, dtype=np.int64)), (ht.UInt64HashTable, np.arange(1000, dtype=np.uint64))])
def test_hashtable_factorize(self, htable, writable, data):
    s = Series(data)
    if htable == ht.Float64HashTable:
        s.loc[500] = np.nan
    elif htable == ht.PyObjectHashTable:
        s.loc[500:502] = [np.nan, None, NaT]
    s_duplicated = s.sample(frac=3, replace=True).reset_index(drop=True)
    s_duplicated.values.setflags(write=writable)
    na_mask = s_duplicated.isna().values
    result_unique, result_inverse = htable().factorize(s_duplicated.values)
    expected_unique = s_duplicated.dropna().drop_duplicates().values
    tm.assert_numpy_array_equal(result_unique, expected_unique)
    result_reconstruct = result_unique[result_inverse[~na_mask]]
    expected_reconstruct = s_duplicated.dropna().values
    tm.assert_numpy_array_equal(result_reconstruct, expected_reconstruct)