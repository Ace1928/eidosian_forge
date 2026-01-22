import itertools
import cirq
import cirq_ft
import numpy as np
import pytest
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@allow_deprecated_cirq_ft_use_in_tests
def test_less_than_gate():
    qubits = cirq.LineQubit.range(4)
    gate = cirq_ft.LessThanGate(3, 5)
    op = gate.on(*qubits)
    circuit = cirq.Circuit(op)
    basis_map = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4, 6: 7, 7: 6, 8: 9, 9: 8, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15}
    cirq.testing.assert_equivalent_computational_basis_map(basis_map, circuit)
    circuit += op ** (-1)
    cirq.testing.assert_equivalent_computational_basis_map(identity_map(len(qubits)), circuit)
    gate2 = cirq_ft.LessThanGate(4, 10)
    assert gate.with_registers(*gate2.registers()) == gate2
    assert cirq.circuit_diagram_info(gate).wire_symbols == ('In(x)',) * 3 + ('+(x < 5)',)
    assert gate ** 1 is gate and gate ** (-1) is gate
    assert gate.__pow__(2) is NotImplemented