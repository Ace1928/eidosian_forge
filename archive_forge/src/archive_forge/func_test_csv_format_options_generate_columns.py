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
def test_csv_format_options_generate_columns(tempdir, dataset_reader):
    path = str(tempdir / 'test.csv')
    with open(path, 'w') as sink:
        sink.write('1,a,true,1\n')
    dataset = ds.dataset(path, format=ds.CsvFileFormat(read_options=pa.csv.ReadOptions(autogenerate_column_names=True)))
    result = dataset_reader.to_table(dataset)
    expected_column_names = ['f0', 'f1', 'f2', 'f3']
    assert result.column_names == expected_column_names
    assert result.equals(pa.table({'f0': pa.array([1]), 'f1': pa.array(['a']), 'f2': pa.array([True]), 'f3': pa.array([1])}))