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
@pytest.mark.parametrize('format', ('ipc', 'parquet'))
def test_read_table_nested_columns(tempdir, format):
    if format == 'parquet':
        pytest.importorskip('pyarrow.parquet')
    table = pa.table({'user_id': ['abc123', 'qrs456'], 'a.dotted.field': [1, 2], 'interaction': [{'type': None, 'element': 'button', 'values': [1, 2], 'structs': [{'foo': 'bar'}, None]}, {'type': 'scroll', 'element': 'window', 'values': [None, 3, 4], 'structs': [{'fizz': 'buzz'}]}]})
    ds.write_dataset(table, tempdir / 'table', format=format)
    ds1 = ds.dataset(tempdir / 'table', format=format)
    table = ds1.to_table(columns=['user_id', 'interaction.type', 'interaction.values', 'interaction.structs', 'a.dotted.field'])
    assert table.to_pylist() == [{'user_id': 'abc123', 'type': None, 'values': [1, 2], 'structs': [{'fizz': None, 'foo': 'bar'}, None], 'a.dotted.field': 1}, {'user_id': 'qrs456', 'type': 'scroll', 'values': [None, 3, 4], 'structs': [{'fizz': 'buzz', 'foo': None}], 'a.dotted.field': 2}]