from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_sizeof():
    schema = pa.schema([pa.field('foo', pa.int32()), pa.field('bar', pa.string())])
    assert sys.getsizeof(schema) > 30
    schema2 = schema.with_metadata({'key': 'some metadata'})
    assert sys.getsizeof(schema2) > sys.getsizeof(schema)
    schema3 = schema.with_metadata({'key': 'some more metadata'})
    assert sys.getsizeof(schema3) > sys.getsizeof(schema2)