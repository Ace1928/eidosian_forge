from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('func', ['nanmean', 'nansum'])
def test_check_bottleneck_disallow(any_real_numpy_dtype, func):
    assert not nanops._bn_ok_dtype(np.dtype(any_real_numpy_dtype).type, func)