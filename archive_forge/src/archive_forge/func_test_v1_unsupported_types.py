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
def test_v1_unsupported_types():
    table = pa.table([pa.array([[1, 2, 3], [], None])], names=['f0'])
    buf = io.BytesIO()
    with pytest.raises(TypeError, match='Unsupported Feather V1 type: list<item: int64>. Use V2 format to serialize all Arrow types.'):
        write_feather(table, buf, version=1)