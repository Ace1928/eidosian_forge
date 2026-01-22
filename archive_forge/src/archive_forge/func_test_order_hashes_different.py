import re
import weakref
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.dtypes import NpyDatetimeUnit
from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.sparse import SparseArray
@pytest.mark.parametrize('v1, v2', [([1, 2, 3], [1, 2, 3]), ([1, 2, 3], [3, 2, 1])])
def test_order_hashes_different(self, v1, v2):
    c1 = CategoricalDtype(v1, ordered=False)
    c2 = CategoricalDtype(v2, ordered=True)
    c3 = CategoricalDtype(v1, ordered=None)
    assert c1 is not c2
    assert c1 is not c3