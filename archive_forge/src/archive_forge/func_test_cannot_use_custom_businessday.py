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
def test_cannot_use_custom_businessday(self):
    msg = 'C is not supported as period frequency'
    msg1 = '<CustomBusinessDay> is not supported as period frequency'
    msg2 = 'PeriodDtype\\[B\\] is deprecated'
    with pytest.raises(ValueError, match=msg):
        PeriodDtype('C')
    with pytest.raises(ValueError, match=msg1):
        with tm.assert_produces_warning(FutureWarning, match=msg2):
            PeriodDtype(pd.offsets.CustomBusinessDay())