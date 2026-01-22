from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import DuplicateKeyError, FieldSelectionError
from petl.test.helpers import eq_
from petl import cut, lookup, lookupone, dictlookup, dictlookupone, \
def test_recordlookup():
    t1 = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 3))
    lkp = recordlookup(t1, 'foo')
    eq_(['a'], [r.foo for r in lkp['a']])
    eq_(['b', 'b'], [r.foo for r in lkp['b']])
    eq_([1], [r.bar for r in lkp['a']])
    eq_([2, 3], [r.bar for r in lkp['b']])
    lkp = recordlookup(cut(t1, 'foo'), 'foo')
    eq_(['a'], [r.foo for r in lkp['a']])
    eq_(['b', 'b'], [r.foo for r in lkp['b']])