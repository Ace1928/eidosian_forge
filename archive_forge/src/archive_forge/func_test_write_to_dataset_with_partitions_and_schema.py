import datetime
import inspect
import os
import pathlib
import numpy as np
import pytest
import unittest.mock as mock
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow import fs
from pyarrow.filesystem import LocalFileSystem
from pyarrow.tests import util
from pyarrow.util import guid
from pyarrow.vendored.version import Version
@pytest.mark.pandas
def test_write_to_dataset_with_partitions_and_schema(tempdir):
    schema = pa.schema([pa.field('group1', type=pa.string()), pa.field('group2', type=pa.string()), pa.field('num', type=pa.int64()), pa.field('nan', type=pa.int32()), pa.field('date', type=pa.timestamp(unit='us'))])
    _test_write_to_dataset_with_partitions(str(tempdir), schema=schema)