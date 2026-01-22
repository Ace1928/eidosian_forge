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
@pytest.mark.parametrize('file', ['stata6_113', 'stata6_114', 'stata6_115', 'stata6_117'])
def test_read_write_reread_dta15(self, file, datapath):
    expected = self.read_csv(datapath('io', 'data', 'stata', 'stata6.csv'))
    expected['byte_'] = expected['byte_'].astype(np.int8)
    expected['int_'] = expected['int_'].astype(np.int16)
    expected['long_'] = expected['long_'].astype(np.int32)
    expected['float_'] = expected['float_'].astype(np.float32)
    expected['double_'] = expected['double_'].astype(np.float64)
    expected['date_td'] = expected['date_td'].apply(datetime.strptime, args=('%Y-%m-%d',))
    file = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = self.read_dta(file)
    tm.assert_frame_equal(expected, parsed)