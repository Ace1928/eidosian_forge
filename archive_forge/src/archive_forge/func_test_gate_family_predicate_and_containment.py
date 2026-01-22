from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np
@pytest.mark.parametrize('gate_family, gates_to_check', [(cirq.GateFamily(CustomXPowGate), [(CustomX, True), (CustomX ** 0.5, True), (CustomX ** sympy.Symbol('theta'), True), (CustomXPowGate(exponent=0.25, global_shift=0.15), True), (cirq.testing.SingleQubitGate(), False), (cirq.X ** 0.5, False), (None, False), (cirq.global_phase_operation(1j), False)]), (cirq.GateFamily(CustomX), [(CustomX, True), (CustomXPowGate(exponent=1, global_shift=0.15), True), (CustomX ** 2, False), (CustomX ** 3, True), (CustomX ** sympy.Symbol('theta'), False), (None, False), (cirq.global_phase_operation(1j), False)]), (cirq.GateFamily(CustomX, ignore_global_phase=False), [(CustomX, True), (CustomXPowGate(exponent=1, global_shift=0.15), False)])])
def test_gate_family_predicate_and_containment(gate_family, gates_to_check):
    for gate, result in gates_to_check:
        assert gate_family._predicate(gate) == result
        assert (gate in gate_family) == result
        if isinstance(gate, cirq.Gate):
            assert (gate(q) in gate_family) == result
            assert (gate(q).with_tags('tags') in gate_family) == result