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
def test_fragments_parquet_row_groups_predicate(tempdir):
    table, dataset = _create_dataset_for_fragments(tempdir, chunk_size=2)
    fragment = list(dataset.get_fragments())[0]
    assert fragment.partition_expression.equals(ds.field('part') == 'a')
    row_group_fragments = list(fragment.split_by_row_group(filter=ds.field('part') == 'a', schema=dataset.schema))
    assert len(row_group_fragments) == 2
    row_group_fragments = list(fragment.split_by_row_group(filter=ds.field('part') == 'b', schema=dataset.schema))
    assert len(row_group_fragments) == 0