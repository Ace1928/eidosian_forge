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
def test_minimal_size_col(self):
    str_lens = (1, 100, 244)
    s = {}
    for str_len in str_lens:
        s['s' + str(str_len)] = Series(['a' * str_len, 'b' * str_len, 'c' * str_len])
    original = DataFrame(s)
    with tm.ensure_clean() as path:
        original.to_stata(path, write_index=False)
        with StataReader(path) as sr:
            sr._ensure_open()
            for variable, fmt, typ in zip(sr._varlist, sr._fmtlist, sr._typlist):
                assert int(variable[1:]) == int(fmt[1:-1])
                assert int(variable[1:]) == typ