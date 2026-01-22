import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_feather_options(fsspectest):
    pytest.importorskip('pyarrow')
    df = DataFrame({'a': [0]})
    df.to_feather('testmem://mockfile', storage_options={'test': 'feather_write'})
    assert fsspectest.test[0] == 'feather_write'
    out = read_feather('testmem://mockfile', storage_options={'test': 'feather_read'})
    assert fsspectest.test[0] == 'feather_read'
    tm.assert_frame_equal(df, out)