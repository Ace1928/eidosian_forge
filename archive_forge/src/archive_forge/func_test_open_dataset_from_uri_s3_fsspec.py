import contextlib
import datetime
import os
import pathlib
import posixpath
import sys
import tempfile
import textwrap
import threading
import time
from shutil import copytree
from urllib.parse import quote
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv
import pyarrow.feather
import pyarrow.fs as fs
import pyarrow.json
from pyarrow.tests.util import (FSProtocolClass, ProxyHandler,
@pytest.mark.parquet
@pytest.mark.s3
def test_open_dataset_from_uri_s3_fsspec(s3_example_simple):
    table, path, _, _, host, port, access_key, secret_key = s3_example_simple
    s3fs = pytest.importorskip('s3fs')
    from pyarrow.fs import FSSpecHandler, PyFileSystem
    fs = s3fs.S3FileSystem(key=access_key, secret=secret_key, client_kwargs={'endpoint_url': 'http://{}:{}'.format(host, port)})
    dataset = ds.dataset(path, format='parquet', filesystem=fs)
    assert dataset.to_table().equals(table)
    fs = PyFileSystem(FSSpecHandler(fs))
    dataset = ds.dataset(path, format='parquet', filesystem=fs)
    assert dataset.to_table().equals(table)