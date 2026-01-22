from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('klass,dtype,na_val', [(Index, np.float64, np.nan), (DatetimeIndex, 'datetime64[ns]', pd.NaT)])
def test_index_ctor_infer_nan_nat(self, klass, dtype, na_val):
    na_list = [na_val, na_val]
    expected = klass(na_list)
    assert expected.dtype == dtype
    result = Index(na_list)
    tm.assert_index_equal(result, expected)
    result = Index(np.array(na_list))
    tm.assert_index_equal(result, expected)