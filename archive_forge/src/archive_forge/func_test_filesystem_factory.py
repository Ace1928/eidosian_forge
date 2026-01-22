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
@pytest.mark.parametrize('paths_or_selector', [fs.FileSelector('subdir', recursive=True), ['subdir/1/xxx/file0.parquet', 'subdir/2/yyy/file1.parquet']])
@pytest.mark.parametrize('pre_buffer', [False, True])
@pytest.mark.parquet
def test_filesystem_factory(mockfs, paths_or_selector, pre_buffer):
    format = ds.ParquetFileFormat(read_options=ds.ParquetReadOptions(dictionary_columns={'str'}), pre_buffer=pre_buffer)
    options = ds.FileSystemFactoryOptions('subdir')
    options.partitioning = ds.DirectoryPartitioning(pa.schema([pa.field('group', pa.int32()), pa.field('key', pa.string())]))
    assert options.partition_base_dir == 'subdir'
    assert options.selector_ignore_prefixes == ['.', '_']
    assert options.exclude_invalid_files is False
    factory = ds.FileSystemDatasetFactory(mockfs, paths_or_selector, format, options)
    inspected_schema = factory.inspect()
    assert factory.inspect().equals(pa.schema([pa.field('i64', pa.int64()), pa.field('f64', pa.float64()), pa.field('str', pa.dictionary(pa.int32(), pa.string())), pa.field('const', pa.int64()), pa.field('struct', pa.struct({'a': pa.int64(), 'b': pa.string()})), pa.field('group', pa.int32()), pa.field('key', pa.string())]), check_metadata=False)
    assert isinstance(factory.inspect_schemas(), list)
    assert isinstance(factory.finish(inspected_schema), ds.FileSystemDataset)
    assert factory.root_partition.equals(ds.scalar(True))
    dataset = factory.finish()
    assert isinstance(dataset, ds.FileSystemDataset)
    scanner = dataset.scanner()
    expected_i64 = pa.array([0, 1, 2, 3, 4], type=pa.int64())
    expected_f64 = pa.array([0, 1, 2, 3, 4], type=pa.float64())
    expected_str = pa.DictionaryArray.from_arrays(pa.array([0, 1, 2, 3, 4], type=pa.int32()), pa.array('0 1 2 3 4'.split(), type=pa.string()))
    expected_struct = pa.array([{'a': i % 3, 'b': str(i % 3)} for i in range(5)])
    iterator = scanner.scan_batches()
    for (batch, fragment), group, key in zip(iterator, [1, 2], ['xxx', 'yyy']):
        expected_group = pa.array([group] * 5, type=pa.int32())
        expected_key = pa.array([key] * 5, type=pa.string())
        expected_const = pa.array([group - 1] * 5, type=pa.int64())
        assert fragment.partition_expression is not None
        assert batch.num_columns == 7
        assert batch[0].equals(expected_i64)
        assert batch[1].equals(expected_f64)
        assert batch[2].equals(expected_str)
        assert batch[3].equals(expected_const)
        assert batch[4].equals(expected_struct)
        assert batch[5].equals(expected_group)
        assert batch[6].equals(expected_key)
    table = dataset.to_table()
    assert isinstance(table, pa.Table)
    assert len(table) == 10
    assert table.num_columns == 7