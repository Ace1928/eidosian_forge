from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_transpose_empty():
    table1 = (('id', 'colour'),)
    table2 = transpose(table1)
    expect2 = (('id',), ('colour',))
    ieq(expect2, table2)