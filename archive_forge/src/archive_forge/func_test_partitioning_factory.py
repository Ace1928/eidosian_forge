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
@pytest.mark.parametrize('pickled', [lambda x, m: x, lambda x, m: m.loads(m.dumps(x))])
def test_partitioning_factory(mockfs, pickled, pickle_module):
    paths_or_selector = fs.FileSelector('subdir', recursive=True)
    format = ds.ParquetFileFormat()
    options = ds.FileSystemFactoryOptions('subdir')
    partitioning_factory = ds.DirectoryPartitioning.discover(['group', 'key'])
    partitioning_factory = pickled(partitioning_factory, pickle_module)
    assert isinstance(partitioning_factory, ds.PartitioningFactory)
    options.partitioning_factory = partitioning_factory
    factory = ds.FileSystemDatasetFactory(mockfs, paths_or_selector, format, options)
    inspected_schema = factory.inspect()
    expected_schema = pa.schema([('i64', pa.int64()), ('f64', pa.float64()), ('str', pa.string()), ('const', pa.int64()), ('struct', pa.struct({'a': pa.int64(), 'b': pa.string()})), ('group', pa.int32()), ('key', pa.string())])
    assert inspected_schema.equals(expected_schema)
    hive_partitioning_factory = ds.HivePartitioning.discover()
    assert isinstance(hive_partitioning_factory, ds.PartitioningFactory)