from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('nan_op,np_op', [(nanops.nanany, np.any), (nanops.nanall, np.all)])
def test_nan_funcs(self, nan_op, np_op, skipna):
    self.check_funs(nan_op, np_op, skipna, allow_all_nan=False, allow_date=False)