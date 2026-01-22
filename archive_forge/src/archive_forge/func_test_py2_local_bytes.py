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
def test_py2_local_bytes(tmpdir):
    fn = str(tmpdir / 'myfile.txt.gz')
    with gzip.open(fn, mode='wb') as f:
        f.write(b'hello\nworld')
    files = open_files(fn, compression='gzip', mode='rt')
    with files[0] as f:
        assert all((isinstance(line, str) for line in f))