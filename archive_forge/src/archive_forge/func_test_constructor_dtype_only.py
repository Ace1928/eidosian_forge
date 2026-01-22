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
@pytest.mark.parametrize('dtype', ['f8', 'i8', 'M8[ns]', 'm8[ns]', 'category', 'object', 'datetime64[ns, UTC]'])
@pytest.mark.parametrize('index', [None, Index([])])
def test_constructor_dtype_only(self, dtype, index):
    result = Series(dtype=dtype, index=index)
    assert result.dtype == dtype
    assert len(result) == 0