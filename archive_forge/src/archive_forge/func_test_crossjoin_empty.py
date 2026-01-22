from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_crossjoin_empty():
    table1 = (('id', 'colour'), (1, 'blue'), (2, 'red'))
    table2 = (('id', 'shape'),)
    table3 = crossjoin(table1, table2)
    expect3 = (('id', 'colour', 'id', 'shape'),)
    ieq(expect3, table3)