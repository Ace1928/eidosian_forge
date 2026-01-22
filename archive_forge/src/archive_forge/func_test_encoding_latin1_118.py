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
def test_encoding_latin1_118(self, datapath):
    msg = '\nOne or more strings in the dta file could not be decoded using utf-8, and\nso the fallback encoding of latin-1 is being used.  This can happen when a file\nhas been incorrectly encoded by Stata or some other software. You should verify\nthe string values returned are correct.'
    path = datapath('io', 'data', 'stata', 'stata1_encoding_118.dta')
    with tm.assert_produces_warning(UnicodeWarning, filter_level='once') as w:
        encoded = read_stata(path)
        assert len(w) == 1
        assert w[0].message.args[0] == msg
    expected = DataFrame([['Düsseldorf']] * 151, columns=['kreis1849'])
    tm.assert_frame_equal(encoded, expected)