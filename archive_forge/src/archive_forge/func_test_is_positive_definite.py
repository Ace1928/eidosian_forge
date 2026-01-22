from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_is_positive_definite():
    A = DenseMatrix(2, 2, [2, 1, 1, 2])
    assert A.is_positive_definite
    C = DenseMatrix(3, 3, [Symbol('x'), 2, 0, 0, 4, 0, 0, 0, 4])
    assert C.is_positive_definite is None