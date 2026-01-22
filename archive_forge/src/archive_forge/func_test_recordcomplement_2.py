from __future__ import absolute_import, print_function, division
from datetime import datetime
from petl.test.helpers import ieq
from petl.transform.setops import complement, intersection, diff, \
def test_recordcomplement_2():
    tablea = (('foo', 'bar', 'baz'), ('A', 1, True), ('C', 7, False), ('B', 2, False), ('C', 9, True))
    tableb = (('bar', 'foo', 'baz'), (2, 'B', False), (9, 'A', False), (3, 'B', True), (9, 'C', True))
    aminusb = (('foo', 'bar', 'baz'), ('A', 1, True), ('C', 7, False))
    result = recordcomplement(tablea, tableb)
    ieq(aminusb, result)
    bminusa = (('bar', 'foo', 'baz'), (3, 'B', True), (9, 'A', False))
    result = recordcomplement(tableb, tablea)
    ieq(bminusa, result)