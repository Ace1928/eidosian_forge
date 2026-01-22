import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
@pytest.mark.lz4
def test_feather_v017_experimental_compression_backward_compatibility(datadir):
    expected = pa.table({'a': range(5)})
    result = read_table(datadir / 'v0.17.0.version.2-compression.lz4.feather')
    assert result.equals(expected)