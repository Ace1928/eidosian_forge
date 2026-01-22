from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.xfail(reason='failing')
@pytest.mark.parametrize('strict_nan', [True, False])
def test_array_equivalent_nested_dicts(strict_nan):
    left = np.array([{'f1': 1, 'f2': np.array(['a', 'b'], dtype=object)}], dtype=object)
    right = np.array([{'f1': 1, 'f2': np.array(['a', 'b'], dtype=object)}], dtype=object)
    assert array_equivalent(left, right, strict_nan=strict_nan)
    assert not array_equivalent(left, right[::-1], strict_nan=strict_nan)
    right2 = np.array([{'f1': 1, 'f2': ['a', 'b']}], dtype=object)
    assert array_equivalent(left, right2, strict_nan=strict_nan)
    assert not array_equivalent(left, right2[::-1], strict_nan=strict_nan)