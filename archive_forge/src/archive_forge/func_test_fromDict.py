from collections import OrderedDict
from collections.abc import Iterator
from datetime import (
from dateutil.tz import tzoffset
import numpy as np
from numpy import ma
import pytest
from pandas._libs import (
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IntCastingNaNError
import pandas.util._test_decorators as td
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.internals.blocks import NumpyBlock
@pytest.mark.filterwarnings('ignore:elementwise comparison:FutureWarning')
def test_fromDict(self, using_infer_string):
    data = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    series = Series(data)
    tm.assert_is_sorted(series.index)
    data = {'a': 0, 'b': '1', 'c': '2', 'd': datetime.now()}
    series = Series(data)
    assert series.dtype == np.object_
    data = {'a': 0, 'b': '1', 'c': '2', 'd': '3'}
    series = Series(data)
    assert series.dtype == np.object_ if not using_infer_string else 'string'
    data = {'a': '0', 'b': '1'}
    series = Series(data, dtype=float)
    assert series.dtype == np.float64