from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.compat import next
from petl.util.base import header, fieldnames, data, dicts, records, \
def test_values_headerless():
    table = []
    actual = values(table, 'foo')
    with pytest.raises(FieldSelectionError):
        for i in actual:
            pass