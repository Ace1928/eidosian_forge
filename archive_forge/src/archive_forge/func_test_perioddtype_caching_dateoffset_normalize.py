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
def test_perioddtype_caching_dateoffset_normalize(self):
    per_d = PeriodDtype(pd.offsets.YearEnd(normalize=True))
    assert per_d.freq.normalize
    per_d2 = PeriodDtype(pd.offsets.YearEnd(normalize=False))
    assert not per_d2.freq.normalize