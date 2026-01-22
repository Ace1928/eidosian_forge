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
@pytest.mark.parametrize('sort', [True, False])
@pytest.mark.parametrize('data, uniques', [(np.array(['b', 'a', None, 'b'], dtype=object), np.array(['b', 'a'], dtype=object)), (pd.array([2, 1, np.nan, 2], dtype='Int64'), pd.array([2, 1], dtype='Int64'))], ids=['numpy_array', 'extension_array'])
def test_factorize_use_na_sentinel(self, sort, data, uniques):
    codes, uniques = algos.factorize(data, sort=sort, use_na_sentinel=True)
    if sort:
        expected_codes = np.array([1, 0, -1, 1], dtype=np.intp)
        expected_uniques = algos.safe_sort(uniques)
    else:
        expected_codes = np.array([0, 1, -1, 0], dtype=np.intp)
        expected_uniques = uniques
    tm.assert_numpy_array_equal(codes, expected_codes)
    if isinstance(data, np.ndarray):
        tm.assert_numpy_array_equal(uniques, expected_uniques)
    else:
        tm.assert_extension_array_equal(uniques, expected_uniques)