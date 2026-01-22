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
def test_obj_none_preservation(self):
    arr = np.array(['foo', None], dtype=object)
    result = pd.unique(arr)
    expected = np.array(['foo', None], dtype=object)
    tm.assert_numpy_array_equal(result, expected, strict_nan=True)