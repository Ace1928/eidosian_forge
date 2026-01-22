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
def test_codecs_get_writer_reader():
    expected = pd.DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=pd.Index(list('ABCD'), dtype=object), index=pd.Index([f'i-{i}' for i in range(30)], dtype=object))
    with tm.ensure_clean() as path:
        with open(path, 'wb') as handle:
            with codecs.getwriter('utf-8')(handle) as encoded:
                expected.to_csv(encoded)
        with open(path, 'rb') as handle:
            with codecs.getreader('utf-8')(handle) as encoded:
                df = pd.read_csv(encoded, index_col=0)
    tm.assert_frame_equal(expected, df)