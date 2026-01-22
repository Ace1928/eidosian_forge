from __future__ import annotations
from array import array
import bz2
import datetime
import functools
from functools import partial
import gzip
import io
import os
from pathlib import Path
import pickle
import shutil
import tarfile
from typing import Any
import uuid
import zipfile
import numpy as np
import pytest
from pandas.compat import (
from pandas.compat._optional import import_optional_dependency
from pandas.compat.compressors import flatten_buffer
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.io.generate_legacy_storage_files import create_pickle_data
import pandas.io.common as icom
from pandas.tseries.offsets import (
def test_write_infer(self, compression_ext, get_random_path):
    base = get_random_path
    path1 = base + compression_ext
    path2 = base + '.raw'
    compression = self._extension_to_compression.get(compression_ext.lower())
    with tm.ensure_clean(path1) as p1, tm.ensure_clean(path2) as p2:
        df = DataFrame(1.1 * np.arange(120).reshape((30, 4)), columns=Index(list('ABCD'), dtype=object), index=Index([f'i-{i}' for i in range(30)], dtype=object))
        df.to_pickle(p1)
        with tm.decompress_file(p1, compression=compression) as f:
            with open(p2, 'wb') as fh:
                fh.write(f.read())
        df2 = pd.read_pickle(p2, compression=None)
        tm.assert_frame_equal(df, df2)