from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_delete_zero_rows_and_columns():
    """Tests method for deleting rows and columns containing only zeros."""
    A, B, C = symbols('A, B, C')
    m1 = Matrix([[0, 0], [0, 0], [1, 2]])
    m2 = Matrix([[0, 1, 2], [0, 3, 4], [0, 5, 6]])
    m3 = Matrix([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    m4 = Matrix([[1, 0, 2], [0, 0, 0], [3, 0, 4]])
    m5 = Matrix([[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 0, 3], [0, 0, 0, 4]])
    m6 = Matrix([[0, 0, A], [B, 0, 0], [0, 0, C]])
    assert dixon.delete_zero_rows_and_columns(m1) == Matrix([[1, 2]])
    assert dixon.delete_zero_rows_and_columns(m2) == Matrix([[1, 2], [3, 4], [5, 6]])
    assert dixon.delete_zero_rows_and_columns(m3) == Matrix([[1, 2], [3, 4]])
    assert dixon.delete_zero_rows_and_columns(m4) == Matrix([[1, 2], [3, 4]])
    assert dixon.delete_zero_rows_and_columns(m5) == Matrix([[1], [2], [3], [4]])
    assert dixon.delete_zero_rows_and_columns(m6) == Matrix([[0, A], [B, 0], [0, C]])