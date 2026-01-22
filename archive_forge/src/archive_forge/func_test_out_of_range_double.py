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
def test_out_of_range_double(self):
    df = DataFrame({'ColumnOk': [0.0, np.finfo(np.double).eps, 4.49423283715579e+307], 'ColumnTooBig': [0.0, np.finfo(np.double).eps, np.finfo(np.double).max]})
    msg = 'Column ColumnTooBig has a maximum value \\(.+\\) outside the range supported by Stata \\(.+\\)'
    with pytest.raises(ValueError, match=msg):
        with tm.ensure_clean() as path:
            df.to_stata(path)