from __future__ import annotations
import os
import subprocess
import sys
import time
import fsspec
import pytest
from fsspec.core import open_files
from packaging.version import parse as parse_version
import dask.bag as db
from dask.utils import tmpdir
def test_open_glob(dir_server):
    root = 'http://localhost:8999/'
    fs = open_files(root + '*')
    assert fs[0].path == 'http://localhost:8999/a'
    assert fs[1].path == 'http://localhost:8999/b'