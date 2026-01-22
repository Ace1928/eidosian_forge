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
def test_legacy_table_fixed_format_read_datetime_py2(datapath):
    expected = DataFrame([[Timestamp('2020-02-06T18:00')]], columns=['A'], index=Index(['date']), dtype='M8[ns]')
    with ensure_clean_store(datapath('io', 'data', 'legacy_hdf', 'legacy_table_fixed_datetime_py2.h5'), mode='r') as store:
        result = store.select('df')
    tm.assert_frame_equal(expected, result)