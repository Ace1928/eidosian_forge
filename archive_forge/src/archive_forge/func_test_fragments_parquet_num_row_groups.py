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
def test_fragments_parquet_num_row_groups(tempdir):
    table = pa.table({'a': range(8)})
    pq.write_table(table, tempdir / 'test.parquet', row_group_size=2)
    dataset = ds.dataset(tempdir / 'test.parquet', format='parquet')
    original_fragment = list(dataset.get_fragments())[0]
    fragment = original_fragment.format.make_fragment(original_fragment.path, original_fragment.filesystem, row_groups=[1, 3])
    assert fragment.num_row_groups == 2
    fragment.ensure_complete_metadata()
    assert fragment.num_row_groups == 2
    assert len(fragment.row_groups) == 2