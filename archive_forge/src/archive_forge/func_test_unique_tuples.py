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
@pytest.mark.parametrize('arr, uniques', [([(0, 0), (0, 1), (1, 0), (1, 1), (0, 0), (0, 1), (1, 0), (1, 1)], [(0, 0), (0, 1), (1, 0), (1, 1)]), ([('b', 'c'), ('a', 'b'), ('a', 'b'), ('b', 'c')], [('b', 'c'), ('a', 'b')]), ([('a', 1), ('b', 2), ('a', 3), ('a', 1)], [('a', 1), ('b', 2), ('a', 3)])])
def test_unique_tuples(self, arr, uniques):
    expected = np.empty(len(uniques), dtype=object)
    expected[:] = uniques
    msg = 'unique with argument that is not not a Series'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = pd.unique(arr)
    tm.assert_numpy_array_equal(result, expected)