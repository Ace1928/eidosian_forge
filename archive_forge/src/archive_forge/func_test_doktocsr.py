from sympy.matrices.sparsetools import _doktocsr, _csrtodok, banded
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices import SparseMatrix
from sympy.testing.pytest import raises
def test_doktocsr():
    a = SparseMatrix([[1, 2, 0, 0], [0, 3, 9, 0], [0, 1, 4, 0]])
    b = SparseMatrix(4, 6, [10, 20, 0, 0, 0, 0, 0, 30, 0, 40, 0, 0, 0, 0, 50, 60, 70, 0, 0, 0, 0, 0, 0, 80])
    c = SparseMatrix(4, 4, [0, 0, 0, 0, 0, 12, 0, 2, 15, 0, 12, 0, 0, 0, 0, 4])
    d = SparseMatrix(10, 10, {(1, 1): 12, (3, 5): 7, (7, 8): 12})
    e = SparseMatrix([[0, 0, 0], [1, 0, 2], [3, 0, 0]])
    f = SparseMatrix(7, 8, {(2, 3): 5, (4, 5): 12})
    assert _doktocsr(a) == [[1, 2, 3, 9, 1, 4], [0, 1, 1, 2, 1, 2], [0, 2, 4, 6], [3, 4]]
    assert _doktocsr(b) == [[10, 20, 30, 40, 50, 60, 70, 80], [0, 1, 1, 3, 2, 3, 4, 5], [0, 2, 4, 7, 8], [4, 6]]
    assert _doktocsr(c) == [[12, 2, 15, 12, 4], [1, 3, 0, 2, 3], [0, 0, 2, 4, 5], [4, 4]]
    assert _doktocsr(d) == [[12, 7, 12], [1, 5, 8], [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3], [10, 10]]
    assert _doktocsr(e) == [[1, 2, 3], [0, 2, 0], [0, 0, 2, 3], [3, 3]]
    assert _doktocsr(f) == [[5, 12], [3, 5], [0, 0, 0, 1, 1, 2, 2, 2], [7, 8]]