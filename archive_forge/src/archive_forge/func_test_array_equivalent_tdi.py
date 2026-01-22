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
@pytest.mark.parametrize('dtype_equal', [True, False])
def test_array_equivalent_tdi(dtype_equal):
    assert array_equivalent(TimedeltaIndex([0, np.nan]), TimedeltaIndex([0, np.nan]), dtype_equal=dtype_equal)
    assert not array_equivalent(TimedeltaIndex([0, np.nan]), TimedeltaIndex([1, np.nan]), dtype_equal=dtype_equal)