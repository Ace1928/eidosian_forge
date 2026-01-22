from sympy.combinatorics.fp_groups import FpGroup
from sympy.combinatorics.coset_table import (CosetTable,
from sympy.combinatorics.coset_table import modified_coset_enumeration_r
from sympy.combinatorics.free_groups import free_group
from sympy.testing.pytest import slow
def test_scan_1():
    F, x, y = free_group('x, y')
    f = FpGroup(F, [x ** 3, y ** 3, x ** (-1) * y ** (-1) * x * y])
    c = CosetTable(f, [x])
    c.scan_and_fill(0, x)
    assert c.table == [[0, 0, None, None]]
    assert c.p == [0]
    assert c.n == 1
    assert c.omega == [0]
    c.scan_and_fill(0, x ** 3)
    assert c.table == [[0, 0, None, None]]
    assert c.p == [0]
    assert c.n == 1
    assert c.omega == [0]
    c.scan_and_fill(0, y ** 3)
    assert c.table == [[0, 0, 1, 2], [None, None, 2, 0], [None, None, 0, 1]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    c.scan_and_fill(0, x ** (-1) * y ** (-1) * x * y)
    assert c.table == [[0, 0, 1, 2], [None, None, 2, 0], [2, 2, 0, 1]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    c.scan_and_fill(1, x ** 3)
    assert c.table == [[0, 0, 1, 2], [3, 4, 2, 0], [2, 2, 0, 1], [4, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]
    c.scan_and_fill(1, y ** 3)
    assert c.table == [[0, 0, 1, 2], [3, 4, 2, 0], [2, 2, 0, 1], [4, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]
    c.scan_and_fill(1, x ** (-1) * y ** (-1) * x * y)
    assert c.table == [[0, 0, 1, 2], [1, 1, 2, 0], [2, 2, 0, 1], [None, 1, None, None], [1, 3, None, None]]
    assert c.p == [0, 1, 2, 1, 1]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    f = FpGroup(F, [x ** 2, y ** 3, (x * y) ** 3])
    c = CosetTable(f, [x * y])
    c.scan_and_fill(0, x * y)
    assert c.table == [[1, None, None, 1], [None, 0, 0, None]]
    assert c.p == [0, 1]
    assert c.n == 2
    assert c.omega == [0, 1]
    c.scan_and_fill(0, x ** 2)
    assert c.table == [[1, 1, None, 1], [0, 0, 0, None]]
    assert c.p == [0, 1]
    assert c.n == 2
    assert c.omega == [0, 1]
    c.scan_and_fill(0, y ** 3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    c.scan_and_fill(0, (x * y) ** 3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    c.scan_and_fill(1, x ** 2)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    c.scan_and_fill(1, y ** 3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [None, None, 1, 0]]
    assert c.p == [0, 1, 2]
    assert c.n == 3
    assert c.omega == [0, 1, 2]
    c.scan_and_fill(1, (x * y) ** 3)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [3, 4, 1, 0], [None, 2, 4, None], [2, None, None, 3]]
    assert c.p == [0, 1, 2, 3, 4]
    assert c.n == 5
    assert c.omega == [0, 1, 2, 3, 4]
    c.scan_and_fill(2, x ** 2)
    assert c.table == [[1, 1, 2, 1], [0, 0, 0, 2], [3, 3, 1, 0], [2, 2, 3, 3], [2, None, None, 3]]
    assert c.p == [0, 1, 2, 3, 3]
    assert c.n == 4
    assert c.omega == [0, 1, 2, 3]