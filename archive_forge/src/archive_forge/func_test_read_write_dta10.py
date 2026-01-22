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
def test_read_write_dta10(self, version):
    original = DataFrame(data=[['string', 'object', 1, 1.1, np.datetime64('2003-12-25')]], columns=['string', 'object', 'integer', 'floating', 'datetime'])
    original['object'] = Series(original['object'], dtype=object)
    original.index.name = 'index'
    original.index = original.index.astype(np.int32)
    original['integer'] = original['integer'].astype(np.int32)
    with tm.ensure_clean() as path:
        original.to_stata(path, convert_dates={'datetime': 'tc'}, version=version)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), original, check_index_type=False)