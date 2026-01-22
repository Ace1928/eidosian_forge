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
def test_write_iterable(tempdir):
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    base_dir = tempdir / 'inmemory_iterable'
    ds.write_dataset((batch for batch in table.to_batches()), base_dir, schema=table.schema, basename_template='dat_{i}.arrow', format='feather')
    result = ds.dataset(base_dir, format='ipc').to_table()
    assert result.equals(table)
    base_dir = tempdir / 'inmemory_reader'
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    ds.write_dataset(reader, base_dir, basename_template='dat_{i}.arrow', format='feather')
    result = ds.dataset(base_dir, format='ipc').to_table()
    assert result.equals(table)