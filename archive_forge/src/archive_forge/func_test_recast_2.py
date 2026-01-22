from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_recast_2():
    table = (('id', 'variable', 'value'), (3, 'age', 16), (1, 'gender', 'F'), (2, 'gender', 'M'), (2, 'age', 17), (1, 'age', 12), (3, 'gender', 'M'))
    expectation = (('id', 'gender'), (1, 'F'), (2, 'M'), (3, 'M'))
    result = recast(table, key='id', variablefield={'variable': ['gender']})
    ieq(expectation, result)