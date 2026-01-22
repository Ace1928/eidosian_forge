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
def test_array_equivalent_compat():
    m = np.array([(1, 2), (3, 4)], dtype=[('a', int), ('b', float)])
    n = np.array([(1, 2), (3, 4)], dtype=[('a', int), ('b', float)])
    assert array_equivalent(m, n, strict_nan=True)
    assert array_equivalent(m, n, strict_nan=False)
    m = np.array([(1, 2), (3, 4)], dtype=[('a', int), ('b', float)])
    n = np.array([(1, 2), (4, 3)], dtype=[('a', int), ('b', float)])
    assert not array_equivalent(m, n, strict_nan=True)
    assert not array_equivalent(m, n, strict_nan=False)
    m = np.array([(1, 2), (3, 4)], dtype=[('a', int), ('b', float)])
    n = np.array([(1, 2), (3, 4)], dtype=[('b', int), ('a', float)])
    assert not array_equivalent(m, n, strict_nan=True)
    assert not array_equivalent(m, n, strict_nan=False)