from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import DuplicateKeyError, FieldSelectionError
from petl.test.helpers import eq_
from petl import cut, lookup, lookupone, dictlookup, dictlookupone, \
def test_lookupone():
    t1 = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 3))
    try:
        lookupone(t1, 'foo', 'bar', strict=True)
    except DuplicateKeyError:
        pass
    else:
        assert False, 'expected error'
    actual = lookupone(t1, 'foo', 'bar', strict=False)
    expect = {'a': 1, 'b': 2}
    eq_(expect, actual)
    actual = lookupone(t1, 'foo', strict=False)
    expect = {'a': ('a', 1), 'b': ('b', 2)}
    eq_(expect, actual)
    actual = lookupone(cut(t1, 'foo'), 'foo')
    expect = {'a': ('a',), 'b': ('b',)}
    eq_(expect, actual)
    t2 = (('foo', 'bar', 'baz'), ('a', 1, True), ('b', 2, False), ('b', 3, True), ('b', 3, False))
    actual = lookupone(t2, 'foo', ('bar', 'baz'), strict=False)
    expect = {'a': (1, True), 'b': (2, False)}
    eq_(expect, actual)
    actual = lookupone(t2, ('foo', 'bar'), 'baz', strict=False)
    expect = {('a', 1): True, ('b', 2): False, ('b', 3): True}
    eq_(expect, actual)