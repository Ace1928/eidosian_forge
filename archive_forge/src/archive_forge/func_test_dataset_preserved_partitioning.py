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
@pytest.mark.pandas
@pytest.mark.parquet
def test_dataset_preserved_partitioning(tempdir):
    _, path = _create_single_file(tempdir)
    dataset = ds.dataset(path)
    assert isinstance(dataset.partitioning, ds.DirectoryPartitioning)
    full_table, path = _create_partitioned_dataset(tempdir)
    dataset = ds.dataset(path)
    assert isinstance(dataset.partitioning, ds.DirectoryPartitioning)
    dataset = ds.dataset(path, partitioning='hive')
    part = dataset.partitioning
    assert part is not None
    assert isinstance(part, ds.HivePartitioning)
    assert part.schema == pa.schema([('part', pa.int32())])
    assert len(part.dictionaries) == 1
    assert part.dictionaries[0] == pa.array([0, 1, 2], pa.int32())
    part = ds.partitioning(pa.schema([('part', pa.int32())]), flavor='hive')
    assert isinstance(part, ds.HivePartitioning)
    assert len(part.dictionaries) == 1
    assert all((x is None for x in part.dictionaries))
    dataset = ds.dataset(path, partitioning=part)
    part = dataset.partitioning
    assert isinstance(part, ds.HivePartitioning)
    assert part.schema == pa.schema([('part', pa.int32())])
    assert len(part.dictionaries) == 1
    assert all((x is None for x in part.dictionaries))
    dataset = ds.dataset(path, partitioning='hive')
    dataset2 = ds.FileSystemDataset(list(dataset.get_fragments()), schema=dataset.schema, format=dataset.format, filesystem=dataset.filesystem)
    assert dataset2.partitioning is None
    root_path = tempdir / 'data-partitioned-metadata'
    metadata_path, _ = _create_parquet_dataset_partitioned(root_path)
    dataset = ds.parquet_dataset(metadata_path, partitioning='hive')
    part = dataset.partitioning
    assert part is not None
    assert isinstance(part, ds.HivePartitioning)
    assert part.schema == pa.schema([('part', pa.string())])
    assert len(part.dictionaries) == 1
    assert set(part.dictionaries[0].to_pylist()) == {'a', 'b'}