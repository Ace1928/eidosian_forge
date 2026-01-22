import pytest
import cirq
from cirq import circuits
def test_correct_mappings():
    a, b, c = cirq.LineQubit.range(3)
    cirq.testing.assert_equivalent_computational_basis_map(maps={1: 1, 2: 2}, circuit=circuits.Circuit(cirq.IdentityGate(num_qubits=2).on(a, b)))
    cirq.testing.assert_equivalent_computational_basis_map(maps={1: 4, 2: 2, 4: 1}, circuit=circuits.Circuit(cirq.SWAP(a, c), cirq.I(b)))