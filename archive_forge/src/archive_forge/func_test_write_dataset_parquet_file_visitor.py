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
def test_write_dataset_parquet_file_visitor(tempdir):
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    visitor_called = False

    def file_visitor(written_file):
        nonlocal visitor_called
        if written_file.metadata is not None and written_file.metadata.num_columns == 3:
            visitor_called = True
    base_dir = tempdir / 'parquet_dataset'
    ds.write_dataset(table, base_dir, format='parquet', file_visitor=file_visitor)
    assert visitor_called