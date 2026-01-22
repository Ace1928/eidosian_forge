import contextlib
import datetime as dt
import hashlib
import tempfile
import time
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.pytables.common import (
from pandas.io.pytables import (
@pytest.mark.parametrize('start, stop', [(0, 2), (1, 2), (None, None)])
def test_contiguous_mixed_data_table(start, stop, setup_path):
    df = DataFrame({'a': Series([20111010, 20111011, 20111012]), 'b': Series(['ab', 'cd', 'ab'])})
    with ensure_clean_store(setup_path) as store:
        store.append('test_dataset', df)
        result = store.select('test_dataset', start=start, stop=stop)
        tm.assert_frame_equal(df[start:stop], result)