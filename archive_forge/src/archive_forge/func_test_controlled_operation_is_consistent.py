import itertools
import re
from typing import cast, Tuple, Union
import numpy as np
import pytest
import sympy
import cirq
from cirq import protocols
from cirq.type_workarounds import NotImplementedType
@pytest.mark.parametrize('gate, should_decompose_to_target', [(cirq.X(cirq.NamedQubit('q1')), True), (cirq.X(cirq.NamedQubit('q1')) ** 0.5, True), (cirq.rx(np.pi)(cirq.NamedQubit('q1')), True), (cirq.rx(np.pi / 2)(cirq.NamedQubit('q1')), True), (cirq.Z(cirq.NamedQubit('q1')), True), (cirq.H(cirq.NamedQubit('q1')), True), (cirq.CNOT(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')), True), (cirq.SWAP(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')), True), (cirq.CCZ(cirq.NamedQubit('q1'), cirq.NamedQubit('q2'), cirq.NamedQubit('q3')), True), (cirq.ControlledGate(cirq.ControlledGate(cirq.CCZ))(*cirq.LineQubit.range(5)), True), (GateUsingWorkspaceForApplyUnitary()(cirq.NamedQubit('q1')), True), (GateAllocatingNewSpaceForResult()(cirq.NamedQubit('q1')), True), (cirq.MatrixGate(np.kron(*(cirq.unitary(cirq.H),) * 2), qid_shape=(4,)).on(cirq.NamedQid('q', 4)), False), (cirq.MatrixGate(cirq.testing.random_unitary(4, random_state=1234)).on(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')), False), (cirq.XX(cirq.NamedQubit('q1'), cirq.NamedQubit('q2')) ** sympy.Symbol('s'), True), (cirq.DiagonalGate(sympy.symbols('s1, s2')).on(cirq.NamedQubit('q')), False)])
def test_controlled_operation_is_consistent(gate: cirq.GateOperation, should_decompose_to_target: bool):
    cb = cirq.NamedQubit('ctr')
    cgate = cirq.ControlledOperation([cb], gate)
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(cgate, ignore_known_gates=not should_decompose_to_target)
    cgate = cirq.ControlledOperation([cb], gate, control_values=[0])
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(cgate, ignore_known_gates=not should_decompose_to_target or cirq.is_parameterized(gate))
    cgate = cirq.ControlledOperation([cb], gate, control_values=[(0, 1)])
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(cgate, ignore_known_gates=not should_decompose_to_target or cirq.is_parameterized(gate))
    cb3 = cb.with_dimension(3)
    cgate = cirq.ControlledOperation([cb3], gate, control_values=[(0, 2)])
    cirq.testing.assert_implements_consistent_protocols(cgate)
    cirq.testing.assert_decompose_ends_at_default_gateset(cgate)