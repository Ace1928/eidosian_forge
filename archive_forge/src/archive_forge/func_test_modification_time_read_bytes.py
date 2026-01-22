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
def test_modification_time_read_bytes(s3, s3so):
    with s3_context('compress', files):
        _, a = read_bytes('s3://compress/test/accounts.*', anon=True, **s3so)
        _, b = read_bytes('s3://compress/test/accounts.*', anon=True, **s3so)
        assert [aa._key for aa in concat(a)] == [bb._key for bb in concat(b)]
    with s3_context('compress', valmap(double, files)):
        _, c = read_bytes('s3://compress/test/accounts.*', anon=True, **s3so)
    assert [aa._key for aa in concat(a)] != [cc._key for cc in concat(c)]