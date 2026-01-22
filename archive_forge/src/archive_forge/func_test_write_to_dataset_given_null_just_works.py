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
def test_write_to_dataset_given_null_just_works(tempdir):
    schema = pa.schema([pa.field('col', pa.int64()), pa.field('part', pa.dictionary(pa.int32(), pa.string()))])
    table = pa.table({'part': [None, None, 'a', 'a'], 'col': list(range(4))}, schema=schema)
    path = str(tempdir / 'test_dataset')
    pq.write_to_dataset(table, path, partition_cols=['part'])
    actual_table = pq.read_table(tempdir / 'test_dataset')
    assert actual_table.column('part').to_pylist() == table.column('part').to_pylist()
    assert actual_table.column('col').equals(table.column('col'))