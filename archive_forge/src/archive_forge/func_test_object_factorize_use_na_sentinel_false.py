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
@pytest.mark.parametrize('data, expected_codes, expected_uniques', [(['a', None, 'b', 'a'], np.array([0, 1, 2, 0], dtype=np.dtype('intp')), np.array(['a', np.nan, 'b'], dtype=object)), (['a', np.nan, 'b', 'a'], np.array([0, 1, 2, 0], dtype=np.dtype('intp')), np.array(['a', np.nan, 'b'], dtype=object))])
def test_object_factorize_use_na_sentinel_false(self, data, expected_codes, expected_uniques):
    codes, uniques = algos.factorize(np.array(data, dtype=object), use_na_sentinel=False)
    tm.assert_numpy_array_equal(uniques, expected_uniques, strict_nan=True)
    tm.assert_numpy_array_equal(codes, expected_codes, strict_nan=True)