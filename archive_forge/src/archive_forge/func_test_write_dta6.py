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
def test_write_dta6(self, datapath):
    original = self.read_csv(datapath('io', 'data', 'stata', 'stata3.csv'))
    original.index.name = 'index'
    original.index = original.index.astype(np.int32)
    original['year'] = original['year'].astype(np.int32)
    original['quarter'] = original['quarter'].astype(np.int32)
    with tm.ensure_clean() as path:
        original.to_stata(path, convert_dates=None)
        written_and_read_again = self.read_dta(path)
        tm.assert_frame_equal(written_and_read_again.set_index('index'), original, check_index_type=False)