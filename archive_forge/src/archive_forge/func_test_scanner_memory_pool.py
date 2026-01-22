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
def test_scanner_memory_pool(dataset):
    old_pool = pa.default_memory_pool()
    pool = pa.system_memory_pool()
    pa.set_memory_pool(pool)
    try:
        allocated_before = pool.bytes_allocated()
        scanner = ds.Scanner.from_dataset(dataset)
        _ = scanner.to_table()
        assert pool.bytes_allocated() > allocated_before
    finally:
        pa.set_memory_pool(old_pool)