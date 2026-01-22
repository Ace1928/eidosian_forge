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
def test_closed_mismatch(self):
    msg = "'closed' keyword does not match value specified in dtype string"
    with pytest.raises(ValueError, match=msg):
        IntervalDtype('interval[int64, left]', 'right')