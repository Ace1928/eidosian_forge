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
def test_parquet_read_options():
    opts1 = ds.ParquetReadOptions()
    opts2 = ds.ParquetReadOptions(dictionary_columns=['a', 'b'])
    opts3 = ds.ParquetReadOptions(coerce_int96_timestamp_unit='ms')
    assert opts1.dictionary_columns == set()
    assert opts2.dictionary_columns == {'a', 'b'}
    assert opts1.coerce_int96_timestamp_unit == 'ns'
    assert opts3.coerce_int96_timestamp_unit == 'ms'
    assert opts1 == opts1
    assert opts1 != opts2
    assert opts1 != opts3