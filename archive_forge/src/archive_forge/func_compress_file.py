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
def compress_file(self, src_path, dest_path, compression):
    if compression is None:
        shutil.copyfile(src_path, dest_path)
        return
    if compression == 'gzip':
        f = gzip.open(dest_path, 'w')
    elif compression == 'bz2':
        f = bz2.BZ2File(dest_path, 'w')
    elif compression == 'zip':
        with zipfile.ZipFile(dest_path, 'w', compression=zipfile.ZIP_DEFLATED) as f:
            f.write(src_path, os.path.basename(src_path))
    elif compression == 'tar':
        with open(src_path, 'rb') as fh:
            with tarfile.open(dest_path, mode='w') as tar:
                tarinfo = tar.gettarinfo(src_path, os.path.basename(src_path))
                tar.addfile(tarinfo, fh)
    elif compression == 'xz':
        f = get_lzma_file()(dest_path, 'w')
    elif compression == 'zstd':
        f = import_optional_dependency('zstandard').open(dest_path, 'wb')
    else:
        msg = f'Unrecognized compression type: {compression}'
        raise ValueError(msg)
    if compression not in ['zip', 'tar']:
        with open(src_path, 'rb') as fh:
            with f:
                f.write(fh.read())