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
def test_invalid_date_conversion(self):
    dates = [dt.datetime(1999, 12, 31, 12, 12, 12, 12000), dt.datetime(2012, 12, 21, 12, 21, 12, 21000), dt.datetime(1776, 7, 4, 7, 4, 7, 4000)]
    original = DataFrame({'nums': [1.0, 2.0, 3.0], 'strs': ['apple', 'banana', 'cherry'], 'dates': dates})
    with tm.ensure_clean() as path:
        msg = 'convert_dates key must be a column or an integer'
        with pytest.raises(ValueError, match=msg):
            original.to_stata(path, convert_dates={'wrong_name': 'tc'})