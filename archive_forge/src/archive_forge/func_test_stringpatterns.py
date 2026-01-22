from __future__ import absolute_import, print_function, division
from petl.compat import PY2
from petl.test.helpers import ieq, eq_
from petl.util.counting import valuecount, valuecounter, valuecounts, \
def test_stringpatterns():
    table = (('foo', 'bar'), ('Mr. Foo', '123-1254'), ('Mrs. Bar', '234-1123'), ('Mr. Spo', '123-1254'), ('Mr. Baz', '321 1434'), ('Mrs. Baz', '321 1434'), ('Mr. Quux', '123-1254-XX'))
    actual = stringpatterns(table, 'foo')
    expect = (('pattern', 'count', 'frequency'), ('Aa. Aaa', 3, 3.0 / 6), ('Aaa. Aaa', 2, 2.0 / 6), ('Aa. Aaaa', 1, 1.0 / 6))
    ieq(expect, actual)
    actual = stringpatterns(table, 'bar')
    expect = (('pattern', 'count', 'frequency'), ('999-9999', 3, 3.0 / 6), ('999 9999', 2, 2.0 / 6), ('999-9999-AA', 1, 1.0 / 6))
    ieq(expect, actual)