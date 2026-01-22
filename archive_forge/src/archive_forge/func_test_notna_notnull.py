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
@pytest.mark.parametrize('notna_f', [notna, notnull])
def test_notna_notnull(notna_f):
    assert notna_f(1.0)
    assert not notna_f(None)
    assert not notna_f(np.nan)
    msg = 'use_inf_as_na option is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with cf.option_context('mode.use_inf_as_na', False):
            assert notna_f(np.inf)
            assert notna_f(-np.inf)
            arr = np.array([1.5, np.inf, 3.5, -np.inf])
            result = notna_f(arr)
            assert result.all()
    with tm.assert_produces_warning(FutureWarning, match=msg):
        with cf.option_context('mode.use_inf_as_na', True):
            assert not notna_f(np.inf)
            assert not notna_f(-np.inf)
            arr = np.array([1.5, np.inf, 3.5, -np.inf])
            result = notna_f(arr)
            assert result.sum() == 2