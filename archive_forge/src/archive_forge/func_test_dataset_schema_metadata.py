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
def test_dataset_schema_metadata(tempdir, dataset_reader):
    df = pd.DataFrame({'a': [1, 2, 3]})
    path = tempdir / 'test.parquet'
    df.to_parquet(path)
    dataset = ds.dataset(path)
    schema = dataset_reader.to_table(dataset).schema
    projected_schema = dataset_reader.to_table(dataset, columns=['a']).schema
    assert b'pandas' in schema.metadata
    assert schema.equals(projected_schema, check_metadata=True)