from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_loc_to_fail2(self):
    s = Series(dtype=object)
    s.loc[1] = 1
    s.loc['a'] = 2
    with pytest.raises(KeyError, match='^-1$'):
        s.loc[-1]
    msg = f'''\\"None of \\[Index\\(\\[-1, -2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
    with pytest.raises(KeyError, match=msg):
        s.loc[[-1, -2]]
    msg = '\\"None of \\[Index\\(\\[\'4\'\\], dtype=\'object\'\\)\\] are in the \\[index\\]\\"'
    with pytest.raises(KeyError, match=msg):
        s.loc[Index(['4'], dtype=object)]
    s.loc[-1] = 3
    with pytest.raises(KeyError, match='not in index'):
        s.loc[[-1, -2]]
    s['a'] = 2
    msg = f'''\\"None of \\[Index\\(\\[-2\\], dtype='{np.dtype(int)}'\\)\\] are in the \\[index\\]\\"'''
    with pytest.raises(KeyError, match=msg):
        s.loc[[-2]]
    del s['a']
    with pytest.raises(KeyError, match=msg):
        s.loc[[-2]] = 0