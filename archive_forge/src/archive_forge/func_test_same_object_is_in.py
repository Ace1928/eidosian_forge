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
def test_same_object_is_in(self):

    class LikeNan:

        def __eq__(self, other) -> bool:
            return False

        def __hash__(self):
            return 0
    a, b = (LikeNan(), LikeNan())
    msg = 'isin with argument that is not not a Series'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        tm.assert_numpy_array_equal(algos.isin([a], [a]), np.array([True]))
        tm.assert_numpy_array_equal(algos.isin([a], [b]), np.array([False]))