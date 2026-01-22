from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl.transform.unpacks import unpack, unpackdict
def test_unpack_headerless():
    table = []
    with pytest.raises(ArgumentError):
        for i in unpack(table, 'bar', ['baz', 'quux']):
            pass