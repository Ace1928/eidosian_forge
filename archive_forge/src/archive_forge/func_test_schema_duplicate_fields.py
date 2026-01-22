from collections import OrderedDict
import sys
import weakref
import pytest
import numpy as np
import pyarrow as pa
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
def test_schema_duplicate_fields():
    fields = [pa.field('foo', pa.int32()), pa.field('bar', pa.string()), pa.field('foo', pa.list_(pa.int8()))]
    sch = pa.schema(fields)
    assert sch.names == ['foo', 'bar', 'foo']
    assert sch.types == [pa.int32(), pa.string(), pa.list_(pa.int8())]
    assert len(sch) == 3
    assert repr(sch) == 'foo: int32\nbar: string\nfoo: list<item: int8>\n  child 0, item: int8'
    assert sch[0].name == 'foo'
    assert sch[0].type == fields[0].type
    with pytest.warns(FutureWarning):
        assert sch.field_by_name('bar') == fields[1]
    with pytest.warns(FutureWarning):
        assert sch.field_by_name('xxx') is None
    with pytest.warns((UserWarning, FutureWarning)):
        assert sch.field_by_name('foo') is None
    assert sch.get_field_index('foo') == -1
    assert sch.get_all_field_indices('foo') == [0, 2]