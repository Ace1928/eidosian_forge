from sympy.external import import_module
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.gate import (X, Y, Z, H, CNOT,
from sympy.physics.quantum.identitysearch import (generate_gate_rules,
from sympy.testing.pytest import skip
def test_generate_equivalent_ids_1():
    x, y, z, h = create_gate_sequence()
    assert generate_equivalent_ids((x,)) == {(x,)}
    assert generate_equivalent_ids((x, x)) == {(x, x)}
    assert generate_equivalent_ids((x, y)) == {(x, y), (y, x)}
    gate_seq = (x, y, z)
    gate_ids = {(x, y, z), (y, z, x), (z, x, y), (z, y, x), (y, x, z), (x, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids
    gate_ids = {Mul(x, y, z), Mul(y, z, x), Mul(z, x, y), Mul(z, y, x), Mul(y, x, z), Mul(x, z, y)}
    assert generate_equivalent_ids(gate_seq, return_as_muls=True) == gate_ids
    gate_seq = (x, y, z, h)
    gate_ids = {(x, y, z, h), (y, z, h, x), (h, x, y, z), (h, z, y, x), (z, y, x, h), (y, x, h, z), (z, h, x, y), (x, h, z, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids
    gate_seq = (x, y, x, y)
    gate_ids = {(x, y, x, y), (y, x, y, x)}
    assert generate_equivalent_ids(gate_seq) == gate_ids
    cgate_y = CGate((1,), y)
    gate_seq = (y, cgate_y, y, cgate_y)
    gate_ids = {(y, cgate_y, y, cgate_y), (cgate_y, y, cgate_y, y)}
    assert generate_equivalent_ids(gate_seq) == gate_ids
    cnot = CNOT(1, 0)
    cgate_z = CGate((0,), Z(1))
    gate_seq = (cnot, h, cgate_z, h)
    gate_ids = {(cnot, h, cgate_z, h), (h, cgate_z, h, cnot), (h, cnot, h, cgate_z), (cgate_z, h, cnot, h)}
    assert generate_equivalent_ids(gate_seq) == gate_ids