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
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_read_py2_hdf_file_in_py3(datapath):
    expected = DataFrame([1.0, 2, 3], index=pd.PeriodIndex(['2015-01-01', '2015-01-02', '2015-01-05'], freq='B'))
    with ensure_clean_store(datapath('io', 'data', 'legacy_hdf', 'periodindex_0.20.1_x86_64_darwin_2.7.13.h5'), mode='r') as store:
        result = store['p']
    tm.assert_frame_equal(result, expected)