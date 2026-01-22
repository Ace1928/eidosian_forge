from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('gate, result', [(CustomX, True), (CustomX ** 2, True), (CustomXPowGate(exponent=3, global_shift=0.5), True), (CustomX ** 0.5, True), (CustomXPowGate(exponent=0.5, global_shift=0.5), True), (CustomX ** 0.25, False), (CustomX ** sympy.Symbol('theta'), False), (cirq.testing.TwoQubitGate(), True)])
def test_gateset_contains(gate, result):
    assert (gate in gateset) is result
    op = gate(*cirq.LineQubit.range(gate.num_qubits()))
    assert (op in gateset) is result
    assert (op.with_tags('tags') in gateset) is result
    circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit([op] * 5), repetitions=5)
    assert (circuit_op in gateset) is result
    assert circuit_op not in gateset.with_params(unroll_circuit_op=False)