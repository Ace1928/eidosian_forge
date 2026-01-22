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
def test_write_dataset_with_scanner_use_projected_schema(tempdir):
    """
    Ensure the projected schema is used to validate partitions for scanner

    https://issues.apache.org/jira/browse/ARROW-17228
    """
    table = pa.table([pa.array(range(20))], names=['original_column'])
    table_dataset = ds.dataset(table)
    columns = {'renamed_column': ds.field('original_column')}
    scanner = table_dataset.scanner(columns=columns)
    ds.write_dataset(scanner, tempdir, partitioning=['renamed_column'], format='ipc')
    with pytest.raises(KeyError, match="'Column original_column does not exist in schema"):
        ds.write_dataset(scanner, tempdir, partitioning=['original_column'], format='ipc')