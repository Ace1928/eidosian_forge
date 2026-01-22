from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_recast_1():
    table = (('id', 'variable', 'value'), (3, 'age', 16), (1, 'gender', 'F'), (2, 'gender', 'M'), (2, 'age', 17), (1, 'age', 12), (3, 'gender', 'M'))
    expectation = (('id', 'age', 'gender'), (1, 12, 'F'), (2, 17, 'M'), (3, 16, 'M'))
    result = recast(table)
    ieq(expectation, result)
    result = recast(table, variablefield='variable')
    ieq(expectation, result)
    result = recast(table, key='id', variablefield='variable')
    ieq(expectation, result)
    result = recast(table, key='id', variablefield='variable', valuefield='value')
    ieq(expectation, result)