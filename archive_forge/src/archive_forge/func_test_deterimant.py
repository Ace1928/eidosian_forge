from itertools import product
from sympy.core.relational import (Equality, Unequality)
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.integrals.integrals import integrate
from sympy.matrices.dense import (Matrix, eye, zeros)
from sympy.matrices.immutable import ImmutableMatrix
from sympy.matrices import SparseMatrix
from sympy.matrices.immutable import \
from sympy.abc import x, y
from sympy.testing.pytest import raises
def test_deterimant():
    assert ImmutableMatrix(4, 4, lambda i, j: i + j).det() == 0