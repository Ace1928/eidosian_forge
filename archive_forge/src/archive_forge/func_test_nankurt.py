from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nankurt(self, skipna):
    sp_stats = pytest.importorskip('scipy.stats')
    func1 = partial(sp_stats.kurtosis, fisher=True)
    func = partial(self._skew_kurt_wrap, func=func1)
    with np.errstate(invalid='ignore'):
        self.check_funs(nanops.nankurt, func, skipna, allow_complex=False, allow_date=False, allow_tdelta=False)