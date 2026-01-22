from __future__ import absolute_import, print_function, division
from petl.test.helpers import ieq
from petl import join, leftjoin, rightjoin, outerjoin, crossjoin, antijoin, \
def test_outerjoin_multiple():
    table1 = (('id', 'color', 'cost'), (1, 'blue', 12), (1, 'red', 8), (2, 'yellow', 15), (2, 'orange', 5), (3, 'purple', 4), (4, 'chartreuse', 42))
    table2 = (('id', 'shape', 'size'), (1, 'circle', 'big'), (2, 'square', 'tiny'), (2, 'square', 'big'), (3, 'ellipse', 'small'), (3, 'ellipse', 'tiny'), (5, 'didodecahedron', 3.14159265))
    actual = outerjoin(table1, table2, key='id')
    expect = (('id', 'color', 'cost', 'shape', 'size'), (1, 'blue', 12, 'circle', 'big'), (1, 'red', 8, 'circle', 'big'), (2, 'yellow', 15, 'square', 'tiny'), (2, 'yellow', 15, 'square', 'big'), (2, 'orange', 5, 'square', 'tiny'), (2, 'orange', 5, 'square', 'big'), (3, 'purple', 4, 'ellipse', 'small'), (3, 'purple', 4, 'ellipse', 'tiny'), (4, 'chartreuse', 42, None, None), (5, None, None, 'didodecahedron', 3.14159265))
    ieq(expect, actual)