from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_undecodable_metadata():
    data1 = b'abcdef\xff\x00'
    data2 = b'ghijkl\xff\x00'
    schema = pa.schema([pa.field('ints', pa.int16(), metadata={'key': data1})], metadata={'key': data2})
    assert 'abcdef' in str(schema)
    assert 'ghijkl' in str(schema)