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
def test_partitioning_function():
    schema = pa.schema([('year', pa.int16()), ('month', pa.int8())])
    names = ['year', 'month']
    part = ds.partitioning(schema)
    assert isinstance(part, ds.DirectoryPartitioning)
    part = ds.partitioning(schema, dictionaries='infer')
    assert isinstance(part, ds.PartitioningFactory)
    part = ds.partitioning(field_names=names)
    assert isinstance(part, ds.PartitioningFactory)
    with pytest.raises(ValueError):
        ds.partitioning()
    with pytest.raises(ValueError, match='Expected list'):
        ds.partitioning(field_names=schema)
    with pytest.raises(ValueError, match='Cannot specify both'):
        ds.partitioning(schema, field_names=schema)
    part = ds.partitioning(schema, flavor='hive')
    assert isinstance(part, ds.HivePartitioning)
    part = ds.partitioning(schema, dictionaries='infer', flavor='hive')
    assert isinstance(part, ds.PartitioningFactory)
    part = ds.partitioning(flavor='hive')
    assert isinstance(part, ds.PartitioningFactory)
    with pytest.raises(ValueError):
        ds.partitioning(names, flavor='hive')
    with pytest.raises(ValueError, match="Cannot specify 'field_names'"):
        ds.partitioning(field_names=names, flavor='hive')
    with pytest.raises(ValueError):
        ds.partitioning(schema, flavor='unsupported')