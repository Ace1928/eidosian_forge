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
def test_is_matching_na_nan_matches_none(self):
    assert not libmissing.is_matching_na(None, np.nan)
    assert not libmissing.is_matching_na(np.nan, None)
    assert libmissing.is_matching_na(None, np.nan, nan_matches_none=True)
    assert libmissing.is_matching_na(np.nan, None, nan_matches_none=True)