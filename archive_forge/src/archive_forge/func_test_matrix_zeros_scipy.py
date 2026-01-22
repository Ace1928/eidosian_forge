from sympy.core.random import randint
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)
from sympy.physics.quantum.matrixutils import (
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_matrix_zeros_scipy():
    if not np:
        skip('numpy not installed.')
    if not scipy:
        skip('scipy not installed.')
    sci = matrix_zeros(4, 4, format='scipy.sparse')
    assert isinstance(sci, scipy_sparse_matrix)