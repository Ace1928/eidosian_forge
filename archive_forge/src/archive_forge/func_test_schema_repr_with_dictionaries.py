from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_repr_with_dictionaries():
    fields = [pa.field('one', pa.dictionary(pa.int16(), pa.string())), pa.field('two', pa.int32())]
    sch = pa.schema(fields)
    expected = 'one: dictionary<values=string, indices=int16, ordered=0>\ntwo: int32'
    assert repr(sch) == expected