from sympy.core.random import randint
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, ones, zeros)
from sympy.physics.quantum.matrixutils import (
from sympy.external import import_module
from sympy.testing.pytest import skip
def test_to_scipy_sparse():
    if not np:
        skip('numpy not installed.')
    if not scipy:
        skip('scipy not installed.')
    else:
        sparse = scipy.sparse
    result = sparse.csr_matrix([[1, 2], [3, 4]], dtype='complex')
    assert np.linalg.norm((to_scipy_sparse(m) - result).todense()) == 0.0