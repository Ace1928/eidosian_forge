from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import ieq
from petl.errors import FieldSelectionError
from petl.util import fieldnames
from petl.transform.headers import setheader, extendheader, pushheader, skip, \
def test_extendheader_headerless():
    table = []
    actual = extendheader(table, ['foo', 'bar'])
    expect = [('foo', 'bar')]
    ieq(expect, actual)
    ieq(expect, actual)