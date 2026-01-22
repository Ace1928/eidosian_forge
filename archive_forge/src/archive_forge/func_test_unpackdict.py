from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import ArgumentError
from petl.test.helpers import ieq
from petl.transform.unpacks import unpack, unpackdict
def test_unpackdict():
    table1 = (('foo', 'bar'), (1, {'baz': 'a', 'quux': 'b'}), (2, {'baz': 'c', 'quux': 'd'}), (3, {'baz': 'e', 'quux': 'f'}))
    table2 = unpackdict(table1, 'bar')
    expect2 = (('foo', 'baz', 'quux'), (1, 'a', 'b'), (2, 'c', 'd'), (3, 'e', 'f'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table1 = (('foo', 'bar'), (1, {'baz': 'a', 'quux': 'b'}), (2, {'baz': 'c', 'quux': 'd'}), (3, {'baz': 'e', 'quux': 'f'}))
    table2 = unpackdict(table1, 'bar', includeoriginal=True)
    expect2 = (('foo', 'bar', 'baz', 'quux'), (1, {'baz': 'a', 'quux': 'b'}, 'a', 'b'), (2, {'baz': 'c', 'quux': 'd'}, 'c', 'd'), (3, {'baz': 'e', 'quux': 'f'}, 'e', 'f'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table1 = (('foo', 'bar'), (1, {'baz': 'a', 'quux': 'b'}), (2, {'baz': 'c', 'quux': 'd'}), (3, {'baz': 'e', 'quux': 'f'}))
    table2 = unpackdict(table1, 'bar', keys=['quux'])
    expect2 = (('foo', 'quux'), (1, 'b'), (2, 'd'), (3, 'f'))
    ieq(expect2, table2)
    ieq(expect2, table2)
    table1 = (('foo', 'bar'), (1, {'baz': 'a', 'quux': 'b'}), (2, 'foobar'), (3, {'baz': 'e', 'quux': 'f'}))
    table2 = unpackdict(table1, 'bar')
    expect2 = (('foo', 'baz', 'quux'), (1, 'a', 'b'), (2, None, None), (3, 'e', 'f'))
    ieq(expect2, table2)
    ieq(expect2, table2)