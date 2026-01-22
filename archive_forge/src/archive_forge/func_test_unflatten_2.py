from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_unflatten_2():
    inpt = ('A', 1, True, 'C', 7, False, 'B', 2, False, 'C', 9)
    expect1 = (('f0', 'f1', 'f2'), ('A', 1, True), ('C', 7, False), ('B', 2, False), ('C', 9, None))
    actual1 = unflatten(inpt, 3)
    ieq(expect1, actual1)
    ieq(expect1, actual1)