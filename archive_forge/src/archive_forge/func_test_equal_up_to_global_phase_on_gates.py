import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('gate1,gate2,eq_up_to_global_phase', [(cirq.rz(0.3 * np.pi), cirq.Z ** 0.3, True), (cirq.rz(0.3), cirq.Z ** 0.3, False), (cirq.ZZPowGate(global_shift=0.5), cirq.ZZ, True), (cirq.ZPowGate(global_shift=0.5) ** sympy.Symbol('e'), cirq.Z, False), (cirq.Z ** sympy.Symbol('e'), cirq.Z ** sympy.Symbol('f'), False)])
def test_equal_up_to_global_phase_on_gates(gate1, gate2, eq_up_to_global_phase):
    num_qubits1, num_qubits2 = (cirq.num_qubits(g) for g in (gate1, gate2))
    qubits = cirq.LineQubit.range(max(num_qubits1, num_qubits2) + 1)
    op1, op2 = (gate1(*qubits[:num_qubits1]), gate2(*qubits[:num_qubits2]))
    assert cirq.equal_up_to_global_phase(op1, op2) == eq_up_to_global_phase
    op2_on_diff_qubits = gate2(*qubits[1:num_qubits2 + 1])
    assert not cirq.equal_up_to_global_phase(op1, op2_on_diff_qubits)