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
def test_parquet_dataset_factory_roundtrip(tempdir):
    root_path = tempdir / 'test_parquet_dataset'
    table = pa.table({'f1': [0] * 10, 'f2': np.random.randn(10)})
    metadata_collector = []
    pq.write_to_dataset(table, str(root_path), metadata_collector=metadata_collector)
    metadata_path = str(root_path / '_metadata')
    pq.write_metadata(table.schema, metadata_path, metadata_collector=metadata_collector)
    dataset = ds.parquet_dataset(metadata_path)
    assert dataset.schema.equals(table.schema)
    result = dataset.to_table()
    assert result.num_rows == 10