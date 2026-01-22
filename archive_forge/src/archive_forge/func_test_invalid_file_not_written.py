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
def test_invalid_file_not_written(self, version):
    content = 'Here is one __�__ Another one __·__ Another one __½__'
    df = DataFrame([content], columns=['invalid'])
    with tm.ensure_clean() as path:
        msg1 = "'latin-1' codec can't encode character '\\\\ufffd' in position 14: ordinal not in range\\(256\\)"
        msg2 = "'ascii' codec can't decode byte 0xef in position 14: ordinal not in range\\(128\\)"
        with pytest.raises(UnicodeEncodeError, match=f'{msg1}|{msg2}'):
            df.to_stata(path)