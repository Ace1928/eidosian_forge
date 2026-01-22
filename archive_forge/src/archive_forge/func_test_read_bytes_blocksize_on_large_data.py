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
def test_read_bytes_blocksize_on_large_data(s3_with_yellow_tripdata, s3so):
    _, L = read_bytes(f's3://{test_bucket_name}/nyc-taxi/2015/yellow_tripdata_2015-01.csv', blocksize=None, anon=True, **s3so)
    assert len(L) == 1
    _, L = read_bytes(f's3://{test_bucket_name}/nyc-taxi/2014/*.csv', blocksize=None, anon=True, **s3so)
    assert len(L) == 12