from contextlib import closing
from pathlib import Path
import re
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.compat import is_platform_windows
import pandas as pd
from pandas import (
from pandas.tests.io.pytables.common import (
from pandas.util import _test_decorators as td
from pandas.io.pytables import TableIterator
def test_read_hdf_open_store(tmp_path, setup_path):
    df = DataFrame(np.random.default_rng(2).random((4, 5)), index=list('abcd'), columns=list('ABCDE'))
    df.index.name = 'letters'
    df = df.set_index(keys='E', append=True)
    path = tmp_path / setup_path
    df.to_hdf(path, key='df', mode='w')
    direct = read_hdf(path, 'df')
    with HDFStore(path, mode='r') as store:
        indirect = read_hdf(store, 'df')
        tm.assert_frame_equal(direct, indirect)
        assert store.is_open