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
def test_write_table(tempdir):
    table = pa.table([pa.array(range(20)), pa.array(np.random.randn(20)), pa.array(np.repeat(['a', 'b'], 10))], names=['f1', 'f2', 'part'])
    base_dir = tempdir / 'single'
    ds.write_dataset(table, base_dir, basename_template='dat_{i}.arrow', format='feather')
    file_paths = list(base_dir.rglob('*'))
    expected_paths = [base_dir / 'dat_0.arrow']
    assert set(file_paths) == set(expected_paths)
    result = ds.dataset(base_dir, format='ipc').to_table()
    assert result.equals(table)
    base_dir = tempdir / 'partitioned'
    expected_paths = [base_dir / 'part=a', base_dir / 'part=a' / 'dat_0.arrow', base_dir / 'part=b', base_dir / 'part=b' / 'dat_0.arrow']
    visited_paths = []
    visited_sizes = []

    def file_visitor(written_file):
        visited_paths.append(written_file.path)
        visited_sizes.append(written_file.size)
    partitioning = ds.partitioning(pa.schema([('part', pa.string())]), flavor='hive')
    ds.write_dataset(table, base_dir, format='feather', basename_template='dat_{i}.arrow', partitioning=partitioning, file_visitor=file_visitor)
    file_paths = list(base_dir.rglob('*'))
    assert set(file_paths) == set(expected_paths)
    actual_sizes = [os.path.getsize(path) for path in visited_paths]
    assert visited_sizes == actual_sizes
    result = ds.dataset(base_dir, format='ipc', partitioning=partitioning)
    assert result.to_table().equals(table)
    assert len(visited_paths) == 2
    for visited_path in visited_paths:
        assert pathlib.Path(visited_path) in expected_paths