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
@pytest.mark.parametrize('compression', ['bz2', 'gzip', 'lz4', 'zstd'])
def test_csv_format_compressed(tempdir, compression, dataset_reader):
    if not pyarrow.Codec.is_available(compression):
        pytest.skip('{} support is not built'.format(compression))
    table = pa.table({'a': pa.array([1, 2, 3], type='int64'), 'b': pa.array([0.1, 0.2, 0.3], type='float64')})
    filesystem = fs.LocalFileSystem()
    suffix = compression if compression != 'gzip' else 'gz'
    path = str(tempdir / f'test.csv.{suffix}')
    with filesystem.open_output_stream(path, compression=compression) as sink:
        csv_str = table.to_pandas().to_csv(index=False)
        sink.write(csv_str.encode('utf-8'))
    dataset = ds.dataset(path, format=ds.CsvFileFormat())
    result = dataset_reader.to_table(dataset)
    assert result.equals(table)