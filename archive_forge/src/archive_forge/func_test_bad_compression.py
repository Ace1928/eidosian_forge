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
def test_bad_compression():
    with filetexts(files, mode='b'):
        for func in [read_bytes, open_files]:
            with pytest.raises(ValueError):
                sample, values = func('.test.accounts.*', compression='not-found')