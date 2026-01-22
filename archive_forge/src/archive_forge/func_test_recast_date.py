from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_recast_date():
    dt = datetime.now().replace
    table = (('id', 'variable', 'value'), (dt(hour=3), 'age', 16), (dt(hour=1), 'gender', 'F'), (dt(hour=2), 'gender', 'M'), (dt(hour=2), 'age', 17), (dt(hour=1), 'age', 12), (dt(hour=3), 'gender', 'M'))
    expectation = (('id', 'age', 'gender'), (dt(hour=1), 12, 'F'), (dt(hour=2), 17, 'M'), (dt(hour=3), 16, 'M'))
    result = recast(table)
    ieq(expectation, result)
    result = recast(table, variablefield='variable')
    ieq(expectation, result)
    result = recast(table, key='id', variablefield='variable')
    ieq(expectation, result)
    result = recast(table, key='id', variablefield='variable', valuefield='value')
    ieq(expectation, result)