from symengine import symbols, init_printing
from symengine.lib.symengine_wrapper import (DenseMatrix, Symbol, Integer,
from symengine.test_utilities import raises
import unittest
def test_is_real_matrix():
    A = DenseMatrix(2, 2, [1, 2, 3, I])
    assert not A.is_real_matrix
    B = DenseMatrix(1, 1, [Symbol('x')])
    assert B.is_real_matrix is None
    C = DenseMatrix(3, 3, [0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert C.is_real_matrix