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
def test_directory_partitioning_dictionary_key(mockfs):
    schema = pa.schema([pa.field('group', pa.dictionary(pa.int8(), pa.int32())), pa.field('key', pa.dictionary(pa.int8(), pa.string()))])
    part = ds.DirectoryPartitioning.discover(schema=schema)
    dataset = ds.dataset('subdir', format='parquet', filesystem=mockfs, partitioning=part)
    assert dataset.partitioning.schema == schema
    table = dataset.to_table()
    assert table.column('group').type.equals(schema.types[0])
    assert table.column('group').to_pylist() == [1] * 5 + [2] * 5
    assert table.column('key').type.equals(schema.types[1])
    assert table.column('key').to_pylist() == ['xxx'] * 5 + ['yyy'] * 5