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
def test_union_dataset_filesystem_datasets(multisourcefs):
    dataset = ds.dataset([ds.dataset('/plain', filesystem=multisourcefs), ds.dataset('/schema', filesystem=multisourcefs), ds.dataset('/hive', filesystem=multisourcefs)])
    expected_schema = pa.schema([('date', pa.date32()), ('index', pa.int64()), ('value', pa.float64()), ('color', pa.string())])
    assert dataset.schema.equals(expected_schema)
    dataset = ds.dataset([ds.dataset('/plain', filesystem=multisourcefs), ds.dataset('/schema', filesystem=multisourcefs), ds.dataset('/hive', filesystem=multisourcefs, partitioning='hive')])
    expected_schema = pa.schema([('date', pa.date32()), ('index', pa.int64()), ('value', pa.float64()), ('color', pa.string()), ('year', pa.int32()), ('month', pa.int32())])
    assert dataset.schema.equals(expected_schema)