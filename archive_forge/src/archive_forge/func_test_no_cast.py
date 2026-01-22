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
def test_no_cast(self):
    comps = ['ss', 42]
    values = ['42']
    expected = np.array([False, False])
    msg = 'isin with argument that is not not a Series, Index'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        result = algos.isin(comps, values)
    tm.assert_numpy_array_equal(expected, result)