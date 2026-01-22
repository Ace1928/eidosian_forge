from io import BytesIO
import os
import pathlib
import tarfile
import zipfile
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_to_parquet_gcs_new_file(monkeypatch, tmpdir):
    """Regression test for writing to a not-yet-existent GCS Parquet file."""
    pytest.importorskip('fastparquet')
    pytest.importorskip('gcsfs')
    from fsspec import AbstractFileSystem
    df1 = DataFrame({'int': [1, 3], 'float': [2.0, np.nan], 'str': ['t', 's'], 'dt': date_range('2018-06-18', periods=2)})

    class MockGCSFileSystem(AbstractFileSystem):

        def open(self, path, mode='r', *args):
            if 'w' not in mode:
                raise FileNotFoundError
            return open(os.path.join(tmpdir, 'test.parquet'), mode, encoding='utf-8')
    monkeypatch.setattr('gcsfs.GCSFileSystem', MockGCSFileSystem)
    df1.to_parquet('gs://test/test.csv', index=True, engine='fastparquet', compression=None)