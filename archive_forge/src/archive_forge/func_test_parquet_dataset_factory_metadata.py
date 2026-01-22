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
def test_parquet_dataset_factory_metadata(tempdir):
    root_path = tempdir / 'test_parquet_dataset_factory_metadata'
    metadata_path, table = _create_parquet_dataset_partitioned(root_path)
    dataset = ds.parquet_dataset(metadata_path, partitioning='hive')
    assert dataset.schema.equals(table.schema)
    assert b'key' in dataset.schema.metadata
    fragments = list(dataset.get_fragments())
    assert b'key' in fragments[0].physical_schema.metadata