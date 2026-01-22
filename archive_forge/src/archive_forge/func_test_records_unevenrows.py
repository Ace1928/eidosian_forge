from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.compat import next
from petl.util.base import header, fieldnames, data, dicts, records, \
def test_records_unevenrows():
    table = (('foo', 'bar'), ('a', 1, True), ('b',))
    actual = records(table)
    it = iter(actual)
    o = next(it)
    eq_('a', o['foo'])
    eq_(1, o['bar'])
    o = next(it)
    eq_('b', o['foo'])
    eq_(None, o['bar'])
    it = iter(actual)
    o = next(it)
    eq_('a', o.foo)
    eq_(1, o.bar)
    o = next(it)
    eq_('b', o.foo)
    eq_(None, o.bar)