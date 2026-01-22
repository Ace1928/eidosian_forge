import datetime
import decimal
from collections import OrderedDict
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip, make_sample_file
from pyarrow.fs import LocalFileSystem
from pyarrow.tests import util
def test_parquet_metadata_lifetime(tempdir):
    table = pa.table({'a': [1, 2, 3]})
    pq.write_table(table, tempdir / 'test_metadata_segfault.parquet')
    parquet_file = pq.ParquetFile(tempdir / 'test_metadata_segfault.parquet')
    parquet_file.metadata.row_group(0).column(0).statistics