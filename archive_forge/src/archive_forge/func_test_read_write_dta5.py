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
def test_read_write_dta5(self):
    original = DataFrame([(np.nan, np.nan, np.nan, np.nan, np.nan)], columns=['float_miss', 'double_miss', 'byte_miss', 'int_miss', 'long_miss'])
    original.index.name = 'index'
    with tm.ensure_clean() as path:
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
    expected = original.copy()
    expected.index = expected.index.astype(np.int32)
    tm.assert_frame_equal(written_and_read_again.set_index('index'), expected)