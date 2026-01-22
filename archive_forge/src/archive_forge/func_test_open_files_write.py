from __future__ import annotations
import gzip
import os
import pathlib
import sys
from functools import partial
from time import sleep
import cloudpickle
import pytest
from fsspec.compression import compr
from fsspec.core import open_files
from fsspec.implementations.local import LocalFileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.utils import filetexts
@pytest.mark.parametrize('compression_opener', [(None, open), ('gzip', gzip.open)])
def test_open_files_write(tmpdir, compression_opener):
    compression, opener = compression_opener
    tmpdir = str(tmpdir)
    files = open_files(tmpdir, num=2, mode='wb', compression=compression)
    assert len(files) == 2
    assert {f.mode for f in files} == {'wb'}
    for fil in files:
        with fil as f:
            f.write(b'000')
    files = sorted(os.listdir(tmpdir))
    assert files == ['0.part', '1.part']
    with opener(os.path.join(tmpdir, files[0]), 'rb') as f:
        d = f.read()
    assert d == b'000'