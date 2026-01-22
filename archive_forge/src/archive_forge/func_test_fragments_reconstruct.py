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
def test_fragments_reconstruct(tempdir, dataset_reader, pickle_module):
    table, dataset = _create_dataset_for_fragments(tempdir)

    def assert_yields_projected(fragment, row_slice, columns=None, filter=None):
        actual = fragment.to_table(schema=table.schema, columns=columns, filter=filter)
        column_names = columns if columns else table.column_names
        assert actual.column_names == column_names
        expected = table.slice(*row_slice).select(column_names)
        assert actual.equals(expected)
    fragment = list(dataset.get_fragments())[0]
    parquet_format = fragment.format
    pickled_fragment = pickle_module.loads(pickle_module.dumps(fragment))
    assert dataset_reader.to_table(pickled_fragment) == dataset_reader.to_table(fragment)
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression)
    assert dataset_reader.to_table(new_fragment).equals(dataset_reader.to_table(fragment))
    assert_yields_projected(new_fragment, (0, 4))
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression)
    assert_yields_projected(new_fragment, (0, 2), filter=ds.field('f1') < 2)
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression)
    assert_yields_projected(new_fragment, (0, 2), columns=['f1'], filter=ds.field('f1') < 2.0)
    new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression)
    assert_yields_projected(new_fragment, (0, 4), filter=ds.field('part') == 'a')
    pattern = 'No match for FieldRef.Name\\(part\\) in ' + fragment.physical_schema.to_string(False, False, False)
    with pytest.raises(ValueError, match=pattern):
        new_fragment = parquet_format.make_fragment(fragment.path, fragment.filesystem, partition_expression=fragment.partition_expression)
        dataset_reader.to_table(new_fragment, filter=ds.field('part') == 'a')