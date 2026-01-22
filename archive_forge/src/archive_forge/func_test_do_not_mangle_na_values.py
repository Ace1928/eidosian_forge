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
def test_do_not_mangle_na_values(self, unique_nulls_fixture, unique_nulls_fixture2):
    if unique_nulls_fixture is unique_nulls_fixture2:
        return
    a = np.array([unique_nulls_fixture, unique_nulls_fixture2], dtype=object)
    result = pd.unique(a)
    assert result.size == 2
    assert a[0] is unique_nulls_fixture
    assert a[1] is unique_nulls_fixture2