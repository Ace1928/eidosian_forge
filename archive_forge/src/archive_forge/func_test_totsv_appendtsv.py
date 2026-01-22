from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import logging
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, fromtsv, tocsv, appendcsv, totsv, appendtsv
def test_totsv_appendtsv():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False)
    f.close()
    totsv(table, f.name, encoding='ascii', lineterminator='\n')
    with open(f.name, 'rb') as o:
        data = [b'foo\tbar', b'a\t1', b'b\t2', b'c\t2']
        expect = b'\n'.join(data) + b'\n'
        actual = o.read()
        eq_(expect, actual)
    table2 = (('foo', 'bar'), ('d', 7), ('e', 9), ('f', 1))
    appendtsv(table2, f.name, encoding='ascii', lineterminator='\n')
    with open(f.name, 'rb') as o:
        data = [b'foo\tbar', b'a\t1', b'b\t2', b'c\t2', b'd\t7', b'e\t9', b'f\t1']
        expect = b'\n'.join(data) + b'\n'
        actual = o.read()
        eq_(expect, actual)