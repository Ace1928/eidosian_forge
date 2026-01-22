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
@pytest.mark.parametrize('file', ['stata3_113', 'stata3_114', 'stata3_115', 'stata3_117'])
def test_read_dta3(self, file, datapath):
    file = datapath('io', 'data', 'stata', f'{file}.dta')
    parsed = self.read_dta(file)
    expected = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
    expected = expected.astype(np.float32)
    expected['year'] = expected['year'].astype(np.int16)
    expected['quarter'] = expected['quarter'].astype(np.int8)
    tm.assert_frame_equal(parsed, expected)