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
def test_write_dataset_preserve_nullability(tempdir):
    schema_nullable = pa.schema([pa.field('x', pa.int64(), nullable=False), pa.field('y', pa.int64(), nullable=True)])
    arrays = [[1, 2, 3], [None, 5, None]]
    table = pa.Table.from_arrays(arrays, schema=schema_nullable)
    pq.write_to_dataset(table, tempdir / 'nulltest1')
    dataset = ds.dataset(tempdir / 'nulltest1', format='parquet')
    assert dataset.to_table().schema.equals(schema_nullable)
    ds.write_dataset(table, tempdir / 'nulltest2', format='parquet')
    dataset = ds.dataset(tempdir / 'nulltest2', format='parquet')
    assert dataset.to_table().schema.equals(schema_nullable)
    ds.write_dataset([table, table], tempdir / 'nulltest3', format='parquet')
    dataset = ds.dataset(tempdir / 'nulltest3', format='parquet')
    assert dataset.to_table().schema.equals(schema_nullable)