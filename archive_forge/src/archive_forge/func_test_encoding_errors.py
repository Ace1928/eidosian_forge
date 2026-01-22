import codecs
import errno
from functools import partial
from io import (
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
@pytest.mark.parametrize('encoding_errors', [None, 'strict', 'replace'])
@pytest.mark.parametrize('format', ['csv', 'json'])
def test_encoding_errors(encoding_errors, format):
    msg = "'utf-8' codec can't decode byte"
    bad_encoding = b'\xe4'
    if format == 'csv':
        content = b',' + bad_encoding + b'\n' + bad_encoding * 2 + b',' + bad_encoding
        reader = partial(pd.read_csv, index_col=0)
    else:
        content = b'{"' + bad_encoding * 2 + b'": {"' + bad_encoding + b'":"' + bad_encoding + b'"}}'
        reader = partial(pd.read_json, orient='index')
    with tm.ensure_clean() as path:
        file = Path(path)
        file.write_bytes(content)
        if encoding_errors != 'replace':
            with pytest.raises(UnicodeDecodeError, match=msg):
                reader(path, encoding_errors=encoding_errors)
        else:
            df = reader(path, encoding_errors=encoding_errors)
            decoded = bad_encoding.decode(errors=encoding_errors)
            expected = pd.DataFrame({decoded: [decoded]}, index=[decoded * 2])
            tm.assert_frame_equal(df, expected)