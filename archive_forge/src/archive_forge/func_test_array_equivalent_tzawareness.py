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
@pytest.mark.parametrize('lvalue, rvalue', [(fix_now, fix_utcnow), (fix_now.to_datetime64(), fix_utcnow), (fix_now.to_pydatetime(), fix_utcnow), (fix_now, fix_utcnow), (fix_now.to_datetime64(), fix_utcnow.to_pydatetime()), (fix_now.to_pydatetime(), fix_utcnow.to_pydatetime())])
def test_array_equivalent_tzawareness(lvalue, rvalue):
    left = np.array([lvalue], dtype=object)
    right = np.array([rvalue], dtype=object)
    assert not array_equivalent(left, right, strict_nan=True)
    assert not array_equivalent(left, right, strict_nan=False)