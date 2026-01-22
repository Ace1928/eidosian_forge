from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nanskew(self, skipna):
    sp_stats = pytest.importorskip('scipy.stats')
    func = partial(self._skew_kurt_wrap, func=sp_stats.skew)
    with np.errstate(invalid='ignore'):
        self.check_funs(nanops.nanskew, func, skipna, allow_complex=False, allow_date=False, allow_tdelta=False)