from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq, eq_
from petl.compat import next
from petl.util.base import header, fieldnames, data, dicts, records, \
def test_records_errors():
    table = (('foo', 'bar'), ('a', 1), ('b', 2))
    actual = records(table)
    it = iter(actual)
    o = next(it)
    try:
        o['baz']
    except KeyError:
        pass
    else:
        raise Exception('expected exception not raised')
    try:
        o.baz
    except AttributeError:
        pass
    else:
        raise Exception('expected exception not raised')