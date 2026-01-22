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
def test_csv_fragment_options(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'w') as sink:
        sink.write('col0\nfoo\nspam\nMYNULL\n')
    dataset = ds.dataset(path, format='csv')
    convert_options = pyarrow.csv.ConvertOptions(null_values=['MYNULL'], strings_can_be_null=True)
    options = ds.CsvFragmentScanOptions(convert_options=convert_options, read_options=pa.csv.ReadOptions(block_size=2 ** 16))
    result = dataset_reader.to_table(dataset, fragment_scan_options=options)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'spam', None])}))
    csv_format = ds.CsvFileFormat(convert_options=convert_options)
    dataset = ds.dataset(path, format=csv_format)
    result = dataset_reader.to_table(dataset)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'spam', None])}))
    options = ds.CsvFragmentScanOptions()
    result = dataset_reader.to_table(dataset, fragment_scan_options=options)
    assert result.equals(pa.table({'col0': pa.array(['foo', 'spam', 'MYNULL'])}))