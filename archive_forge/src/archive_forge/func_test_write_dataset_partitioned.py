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
@pytest.mark.pandas
def test_write_dataset_partitioned(tempdir):
    directory = tempdir / 'partitioned'
    _ = _create_parquet_dataset_partitioned(directory)
    partitioning = ds.partitioning(flavor='hive')
    dataset = ds.dataset(directory, partitioning=partitioning)
    target = tempdir / 'partitioned-hive-target'
    expected_paths = [target / 'part=a', target / 'part=a' / 'part-0.arrow', target / 'part=b', target / 'part=b' / 'part-0.arrow']
    partitioning_schema = ds.partitioning(pa.schema([('part', pa.string())]), flavor='hive')
    _check_dataset_roundtrip(dataset, str(target), expected_paths, 'f1', target, partitioning=partitioning_schema)
    target = tempdir / 'partitioned-dir-target'
    expected_paths = [target / 'a', target / 'a' / 'part-0.arrow', target / 'b', target / 'b' / 'part-0.arrow']
    partitioning_schema = ds.partitioning(pa.schema([('part', pa.string())]))
    _check_dataset_roundtrip(dataset, str(target), expected_paths, 'f1', target, partitioning=partitioning_schema)