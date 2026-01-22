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
@pytest.mark.pandas
def test_dataset_partitioned_dictionary_type_reconstruct(tempdir, pickle_module):
    table = pa.table({'part': np.repeat(['A', 'B'], 5), 'col': range(10)})
    part = ds.partitioning(table.select(['part']).schema, flavor='hive')
    ds.write_dataset(table, tempdir, partitioning=part, format='feather')
    dataset = ds.dataset(tempdir, format='feather', partitioning=ds.HivePartitioning.discover(infer_dictionary=True))
    expected = pa.table({'col': table['col'], 'part': table['part'].dictionary_encode()})
    assert dataset.to_table().equals(expected)
    fragment = list(dataset.get_fragments())[0]
    assert fragment.to_table(schema=dataset.schema).equals(expected[:5])
    part_expr = fragment.partition_expression
    restored = pickle_module.loads(pickle_module.dumps(dataset))
    assert restored.to_table().equals(expected)
    restored = pickle_module.loads(pickle_module.dumps(fragment))
    assert restored.to_table(schema=dataset.schema).equals(expected[:5])
    assert restored.to_table(schema=dataset.schema).to_pandas().equals(expected[:5].to_pandas())
    assert restored.partition_expression.equals(part_expr)