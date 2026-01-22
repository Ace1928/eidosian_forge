from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import DuplicateKeyError, FieldSelectionError
from petl.test.helpers import eq_
from petl import cut, lookup, lookupone, dictlookup, dictlookupone, \
def test_lookup_headerless():
    table = []
    with pytest.raises(FieldSelectionError):
        lookup(table, 'foo', 'bar')