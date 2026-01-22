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
@pytest.mark.parametrize('encoding', [None, 'utf-8'])
@pytest.mark.parametrize('format', ['csv', 'json'])
def test_codecs_encoding(encoding, format):
    expected = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
    with tm.ensure_clean() as path:
        with codecs.open(path, mode='w', encoding=encoding) as handle:
            getattr(expected, f'to_{format}')(handle)
        with codecs.open(path, mode='r', encoding=encoding) as handle:
            if format == 'csv':
                df = pd.read_csv(handle, index_col=0)
            else:
                df = pd.read_json(handle)
    tm.assert_frame_equal(expected, df)