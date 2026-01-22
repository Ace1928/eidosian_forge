from datetime import timedelta
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
from pandas import (
import pandas._testing as tm
def test_basic_indexing():
    s = Series(np.random.default_rng(2).standard_normal(5), index=['a', 'b', 'a', 'a', 'b'])
    warn_msg = 'Series.__[sg]etitem__ treating keys as positions is deprecated'
    msg = 'index 5 is out of bounds for axis 0 with size 5'
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5]
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5] = 0
    with pytest.raises(KeyError, match="^'c'$"):
        s['c']
    s = s.sort_index()
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5]
    msg = 'index 5 is out of bounds for axis (0|1) with size 5|^5$'
    with pytest.raises(IndexError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=warn_msg):
            s[5] = 0