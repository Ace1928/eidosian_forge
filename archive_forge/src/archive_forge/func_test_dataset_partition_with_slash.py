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
def test_dataset_partition_with_slash(tmpdir):
    from pyarrow import dataset as ds
    path = tmpdir / 'slash-writer-x'
    dt_table = pa.Table.from_arrays([pa.array([1, 2, 3, 4, 5], pa.int32()), pa.array(['experiment/A/f.csv', 'experiment/B/f.csv', 'experiment/A/f.csv', 'experiment/C/k.csv', 'experiment/M/i.csv'], pa.utf8())], ['exp_id', 'exp_meta'])
    ds.write_dataset(data=dt_table, base_dir=path, format='ipc', partitioning=['exp_meta'], partitioning_flavor='hive')
    read_table = ds.dataset(source=path, format='ipc', partitioning='hive', schema=pa.schema([pa.field('exp_id', pa.int32()), pa.field('exp_meta', pa.utf8())])).to_table().combine_chunks()
    assert dt_table == read_table.sort_by('exp_id')
    exp_meta = dt_table.column(1).to_pylist()
    exp_meta = sorted(set(exp_meta))
    encoded_paths = ['exp_meta=' + quote(path, safe='') for path in exp_meta]
    file_paths = sorted(os.listdir(path))
    assert encoded_paths == file_paths