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
def test_deprecate_order(self):
    data = np.array([2 ** 63, 1, 2 ** 63], dtype=np.uint64)
    with pytest.raises(TypeError, match='got an unexpected keyword'):
        algos.factorize(data, order=True)
    with tm.assert_produces_warning(False):
        algos.factorize(data)