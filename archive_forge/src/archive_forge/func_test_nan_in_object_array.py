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
def test_nan_in_object_array(self):
    duplicated_items = ['a', np.nan, 'c', 'c']
    result = pd.unique(np.array(duplicated_items, dtype=object))
    expected = np.array(['a', np.nan, 'c'], dtype=object)
    tm.assert_numpy_array_equal(result, expected)