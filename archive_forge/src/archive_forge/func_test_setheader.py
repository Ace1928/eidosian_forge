from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_setheader():
    table1 = (('foo', 'bar'), ('a', 1), ('b', 2))
    table2 = setheader(table1, ['foofoo', 'barbar'])
    expect2 = (('foofoo', 'barbar'), ('a', 1), ('b', 2))
    ieq(expect2, table2)
    ieq(expect2, table2)