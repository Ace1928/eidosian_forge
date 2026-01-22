from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
from sympy.testing.pytest import skip
def test_is_scalar_sparse_matrix():
    np = import_module('numpy')
    if not np:
        skip('numpy not installed.')
    scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})
    if not scipy:
        skip('scipy not installed.')
    numqubits = 2
    id_only = False
    id_gate = (IdentityGate(1),)
    assert is_scalar_sparse_matrix(id_gate, numqubits, id_only) is True
    x0 = X(0)
    xx_circuit = (x0, x0)
    assert is_scalar_sparse_matrix(xx_circuit, numqubits, id_only) is True
    x1 = X(1)
    y1 = Y(1)
    xy_circuit = (x1, y1)
    assert is_scalar_sparse_matrix(xy_circuit, numqubits, id_only) is False
    z1 = Z(1)
    xyz_circuit = (x1, y1, z1)
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is True
    cnot = CNOT(1, 0)
    cnot_circuit = (cnot, cnot)
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True
    h = H(0)
    hh_circuit = (h, h)
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True
    h1 = H(1)
    xhzh_circuit = (x1, h1, z1, h1)
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True
    id_only = True
    assert is_scalar_sparse_matrix(xhzh_circuit, numqubits, id_only) is True
    assert is_scalar_sparse_matrix(xyz_circuit, numqubits, id_only) is False
    assert is_scalar_sparse_matrix(cnot_circuit, numqubits, id_only) is True
    assert is_scalar_sparse_matrix(hh_circuit, numqubits, id_only) is True