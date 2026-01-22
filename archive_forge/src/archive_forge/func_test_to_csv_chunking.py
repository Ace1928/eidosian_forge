import csv
from io import StringIO
import os
import numpy as np
import pytest
from pandas.errors import ParserError
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.common as com
from pandas.io.common import get_handle
@pytest.mark.parametrize('chunksize', [10000, 50000, 100000])
def test_to_csv_chunking(self, chunksize):
    aa = DataFrame({'A': range(100000)})
    aa['B'] = aa.A + 1.0
    aa['C'] = aa.A + 2.0
    aa['D'] = aa.A + 3.0
    with tm.ensure_clean() as filename:
        aa.to_csv(filename, chunksize=chunksize)
        rs = read_csv(filename, index_col=0)
        tm.assert_frame_equal(rs, aa)