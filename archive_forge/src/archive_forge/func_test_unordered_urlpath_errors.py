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
def test_unordered_urlpath_errors():
    with pytest.raises(TypeError):
        read_bytes({'sets/are.csv', 'unordered/so/they.csv', 'should/not/be.csv', 'allowed.csv'})