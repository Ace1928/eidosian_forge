from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def test_addfieldusingcontext_stateful():
    table1 = (('foo', 'bar'), ('A', 1), ('B', 4), ('C', 5), ('D', 9))
    expect = (('foo', 'bar', 'baz', 'quux'), ('A', 1, 1, 5), ('B', 4, 5, 10), ('C', 5, 10, 19), ('D', 9, 19, 19))

    def upstream(prv, cur, nxt):
        if prv is None:
            return cur.bar
        else:
            return cur.bar + prv.baz

    def downstream(prv, cur, nxt):
        if nxt is None:
            return prv.quux
        elif prv is None:
            return nxt.bar + cur.bar
        else:
            return nxt.bar + prv.quux
    table2 = addfieldusingcontext(table1, 'baz', upstream)
    table3 = addfieldusingcontext(table2, 'quux', downstream)
    ieq(expect, table3)
    ieq(expect, table3)