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
@pytest.mark.parametrize('blocksize', [5.0, '5 B'])
def test_read_bytes_blocksize_types(blocksize):
    with filetexts(files, mode='b'):
        sample, vals = read_bytes('.test.account*', blocksize=blocksize)
        results = compute(*concat(vals))
        ourlines = b''.join(results).split(b'\n')
        testlines = b''.join(files.values()).split(b'\n')
        assert set(ourlines) == set(testlines)