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
def test_convert_strl_name_swap(self):
    original = DataFrame([['a' * 3000, 'A', 'apple'], ['b' * 1000, 'B', 'banana']], columns=['long1' * 10, 'long', 1])
    original.index.name = 'index'
    with tm.assert_produces_warning(InvalidColumnName):
        with tm.ensure_clean() as path:
            original.to_stata(path, convert_strl=['long', 1], version=117)
            reread = self.read_dta(path)
            reread = reread.set_index('index')
            reread.columns = original.columns
            tm.assert_frame_equal(reread, original, check_index_type=False)