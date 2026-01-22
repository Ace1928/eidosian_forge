import datetime
import decimal
import pytest
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.tests import util
def test_map_scalar_as_py_with_custom_field_name():
    """
    Check we can call `MapScalar.as_py` with custom field names

    See https://github.com/apache/arrow/issues/36809
    """
    assert pa.scalar([('foo', 'bar')], pa.map_(pa.string(), pa.string())).as_py() == [('foo', 'bar')]
    assert pa.scalar([('foo', 'bar')], pa.map_(pa.field('custom_key', pa.string(), nullable=False), pa.field('custom_value', pa.string()))).as_py() == [('foo', 'bar')]