import bz2
import datetime as dt
from datetime import datetime
import gzip
import io
import os
import struct
import tarfile
import zipfile
import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import CategoricalDtype
import pandas._testing as tm
from pandas.core.frame import (
from pandas.io.parsers import read_csv
from pandas.io.stata import (
@pytest.mark.parametrize('version', [114, 117, 118, 119, None])
@pytest.mark.parametrize('file', ['stata5_113', 'stata5_114', 'stata5_115', 'stata5_117'])
def test_read_write_reread_dta14(self, file, parsed_114, version, datapath):
    file = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = self.read_dta(file)
    parsed.index.name = 'index'
    tm.assert_frame_equal(parsed_114, parsed)
    with tm.ensure_clean() as path:
        parsed_114.to_stata(path, convert_dates={'date_td': 'td'}, version=version)
        written_and_read_again = self.read_dta(path)
    expected = parsed_114.copy()
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)