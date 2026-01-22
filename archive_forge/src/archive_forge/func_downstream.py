from __future__ import absolute_import, print_function, division
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.util import expr, empty, coalesce
from petl.transform.basics import cut, cat, addfield, rowslice, head, tail, \
def downstream(prv, cur, nxt):
    if nxt is None:
        return prv.quux
    elif prv is None:
        return nxt.bar + cur.bar
    else:
        return nxt.bar + prv.quux