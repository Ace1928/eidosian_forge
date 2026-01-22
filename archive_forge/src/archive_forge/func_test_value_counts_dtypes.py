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
def test_value_counts_dtypes(self):
    msg2 = 'pandas.value_counts is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        result = algos.value_counts(np.array([1, 1.0]))
    assert len(result) == 1
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        result = algos.value_counts(np.array([1, 1.0]), bins=1)
    assert len(result) == 1
    with tm.assert_produces_warning(FutureWarning, match=msg2):
        result = algos.value_counts(Series([1, 1.0, '1']))
    assert len(result) == 2
    msg = 'bins argument only works with numeric data'
    with pytest.raises(TypeError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            algos.value_counts(np.array(['1', 1], dtype=object), bins=1)