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
def test_count_rows(dataset, dataset_reader):
    fragment = next(dataset.get_fragments())
    assert dataset_reader.count_rows(fragment) == 5
    assert dataset_reader.count_rows(fragment, filter=ds.field('i64') == 4) == 1
    assert dataset_reader.count_rows(dataset) == 10
    assert dataset_reader.count_rows(dataset, filter=ds.field('group') == 1) == 5
    assert dataset_reader.count_rows(dataset, filter=ds.field('i64') >= 3) == 4
    assert dataset_reader.count_rows(dataset, filter=ds.field('i64') < 0) == 0