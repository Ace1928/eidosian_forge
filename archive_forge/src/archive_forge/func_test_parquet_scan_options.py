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
def test_parquet_scan_options():
    opts1 = ds.ParquetFragmentScanOptions()
    opts2 = ds.ParquetFragmentScanOptions(buffer_size=4096)
    opts3 = ds.ParquetFragmentScanOptions(buffer_size=2 ** 13, use_buffered_stream=True)
    opts4 = ds.ParquetFragmentScanOptions(buffer_size=2 ** 13, pre_buffer=False)
    opts5 = ds.ParquetFragmentScanOptions(thrift_string_size_limit=123456, thrift_container_size_limit=989284)
    opts6 = ds.ParquetFragmentScanOptions(page_checksum_verification=True)
    cache_opts = pa.CacheOptions(hole_size_limit=2 ** 10, range_size_limit=8 * 2 ** 10, lazy=True)
    opts7 = ds.ParquetFragmentScanOptions(pre_buffer=True, cache_options=cache_opts)
    assert opts1.use_buffered_stream is False
    assert opts1.buffer_size == 2 ** 13
    assert opts1.pre_buffer is True
    assert opts1.thrift_string_size_limit == 100000000
    assert opts1.thrift_container_size_limit == 1000000
    assert opts1.page_checksum_verification is False
    assert opts2.use_buffered_stream is False
    assert opts2.buffer_size == 2 ** 12
    assert opts2.pre_buffer is True
    assert opts3.use_buffered_stream is True
    assert opts3.buffer_size == 2 ** 13
    assert opts3.pre_buffer is True
    assert opts4.use_buffered_stream is False
    assert opts4.buffer_size == 2 ** 13
    assert opts4.pre_buffer is False
    assert opts5.thrift_string_size_limit == 123456
    assert opts5.thrift_container_size_limit == 989284
    assert opts6.page_checksum_verification is True
    assert opts7.pre_buffer is True
    assert opts7.cache_options == cache_opts
    assert opts7.cache_options != opts1.cache_options
    assert opts1 == opts1
    assert opts1 != opts2
    assert opts2 != opts3
    assert opts3 != opts4
    assert opts5 != opts1
    assert opts6 != opts1
    assert opts7 != opts1