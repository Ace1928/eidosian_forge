from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import DuplicateKeyError, FieldSelectionError
from petl.test.helpers import eq_
from petl import cut, lookup, lookupone, dictlookup, dictlookupone, \
def test_dictlookup():
    t1 = (('foo', 'bar'), ('a', 1), ('b', 2), ('b', 3))
    actual = dictlookup(t1, 'foo')
    expect = {'a': [{'foo': 'a', 'bar': 1}], 'b': [{'foo': 'b', 'bar': 2}, {'foo': 'b', 'bar': 3}]}
    eq_(expect, actual)
    actual = dictlookup(cut(t1, 'foo'), 'foo')
    expect = {'a': [{'foo': 'a'}], 'b': [{'foo': 'b'}, {'foo': 'b'}]}
    eq_(expect, actual)
    t2 = (('foo', 'bar', 'baz'), ('a', 1, True), ('b', 2, False), ('b', 3, True), ('b', 3, False))
    actual = dictlookup(t2, ('foo', 'bar'))
    expect = {('a', 1): [{'foo': 'a', 'bar': 1, 'baz': True}], ('b', 2): [{'foo': 'b', 'bar': 2, 'baz': False}], ('b', 3): [{'foo': 'b', 'bar': 3, 'baz': True}, {'foo': 'b', 'bar': 3, 'baz': False}]}
    eq_(expect, actual)