import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
def test_put_compression(setup_path):
    with ensure_clean_store(setup_path) as store:
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)), columns=Index(list('ABCD'), dtype=object), index=date_range('2000-01-01', periods=10, freq='B'))
        store.put('c', df, format='table', complib='zlib')
        tm.assert_frame_equal(store['c'], df)
        msg = 'Compression not supported on Fixed format stores'
        with pytest.raises(ValueError, match=msg):
            store.put('b', df, format='fixed', complib='zlib')