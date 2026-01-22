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
def test_read_bytes_include_path():
    with filetexts(files, mode='b'):
        _, _, paths = read_bytes('.test.accounts.*', include_path=True)
        assert {os.path.split(path)[1] for path in paths} == files.keys()