from __future__ import annotations
import io
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from functools import partial
import pytest
from fsspec.compression import compr
from fsspec.core import get_fs_token_paths, open_files
from s3fs import S3FileSystem as DaskS3FileSystem
from tlz import concat, valmap
from dask import compute
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
def test_get_s3():
    s3 = DaskS3FileSystem(key='key', secret='secret')
    assert s3.key == 'key'
    assert s3.secret == 'secret'
    s3 = DaskS3FileSystem(username='key', password='secret')
    assert s3.key == 'key'
    assert s3.secret == 'secret'
    with pytest.raises(KeyError):
        DaskS3FileSystem(key='key', username='key')
    with pytest.raises(KeyError):
        DaskS3FileSystem(secret='key', password='key')