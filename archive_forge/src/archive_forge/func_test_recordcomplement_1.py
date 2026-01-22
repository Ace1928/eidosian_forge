from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def test_recordcomplement_1():
    table1 = (('foo', 'bar'), ('A', 1), ('B', 2), ('C', 7))
    table2 = (('bar', 'foo'), (9, 'A'), (2, 'B'), (3, 'B'))
    expectation = (('foo', 'bar'), ('A', 1), ('C', 7))
    result = recordcomplement(table1, table2)
    ieq(expectation, result)