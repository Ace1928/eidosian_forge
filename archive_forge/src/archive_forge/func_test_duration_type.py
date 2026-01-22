import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
def test_duration_type():
    arrays = [pa.array([0, 1, 2, 3], type=pa.duration(unit)) for unit in ['s', 'ms', 'us', 'ns']]
    table = pa.Table.from_arrays(arrays, ['d[s]', 'd[ms]', 'd[us]', 'd[ns]'])
    _check_roundtrip(table)