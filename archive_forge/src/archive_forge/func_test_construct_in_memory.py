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
def test_construct_in_memory(dataset_reader):
    batch = pa.RecordBatch.from_arrays([pa.array(range(10))], names=['a'])
    table = pa.Table.from_batches([batch])
    dataset_table = ds.dataset([], format='ipc', schema=pa.schema([])).to_table()
    assert dataset_table == pa.table([])
    for source in (batch, table, [batch], [table]):
        dataset = ds.dataset(source)
        assert dataset_reader.to_table(dataset) == table
        assert len(list(dataset.get_fragments())) == 1
        assert next(dataset.get_fragments()).to_table() == table
        assert pa.Table.from_batches(list(dataset.to_batches())) == table