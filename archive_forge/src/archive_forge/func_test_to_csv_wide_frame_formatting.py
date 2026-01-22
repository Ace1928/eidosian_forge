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
@pytest.mark.slow
def test_to_csv_wide_frame_formatting(self, monkeypatch):
    chunksize = 100
    df = DataFrame(np.random.default_rng(2).standard_normal((1, chunksize + 10)), columns=None, index=None)
    with tm.ensure_clean() as filename:
        with monkeypatch.context() as m:
            m.setattr('pandas.io.formats.csvs._DEFAULT_CHUNKSIZE_CELLS', chunksize)
            df.to_csv(filename, header=False, index=False)
        rs = read_csv(filename, header=None)
    tm.assert_frame_equal(rs, df)