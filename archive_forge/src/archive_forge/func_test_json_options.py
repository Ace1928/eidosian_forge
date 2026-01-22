import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_json_options(fsspectest, compression):
    df = DataFrame({'a': [0]})
    df.to_json('testmem://mockfile', compression=compression, storage_options={'test': 'json_write'})
    assert fsspectest.test[0] == 'json_write'
    out = read_json('testmem://mockfile', compression=compression, storage_options={'test': 'json_read'})
    assert fsspectest.test[0] == 'json_read'
    tm.assert_frame_equal(df, out)