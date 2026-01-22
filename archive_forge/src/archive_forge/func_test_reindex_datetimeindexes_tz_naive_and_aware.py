import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_reindex_datetimeindexes_tz_naive_and_aware():
    idx = date_range('20131101', tz='America/Chicago', periods=7)
    newidx = date_range('20131103', periods=10, freq='h')
    s = Series(range(7), index=idx)
    msg = 'Cannot compare dtypes datetime64\\[ns, America/Chicago\\] and datetime64\\[ns\\]'
    with pytest.raises(TypeError, match=msg):
        s.reindex(newidx, method='ffill')