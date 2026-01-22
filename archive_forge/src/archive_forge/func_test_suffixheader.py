from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_suffixheader():
    table1 = (('foo', 'bar'), (1, 'A'), (2, 'B'))
    expect = (('foo_suf', 'bar_suf'), (1, 'A'), (2, 'B'))
    actual = suffixheader(table1, '_suf')
    ieq(expect, actual)
    ieq(expect, actual)