import numpy as np
import pytest
import sympy
import cirq
def test_operation_commutes_using_overlap_and_unitary():

    class CustomCnotGate(cirq.Gate):

        def num_qubits(self) -> int:
            return 2

        def _unitary_(self):
            return cirq.unitary(cirq.CNOT)
    custom_cnot_gate = CustomCnotGate()

    class CustomCnotOp(cirq.Operation):

        def __init__(self, *qs: cirq.Qid):
            self.qs = qs

        def _unitary_(self):
            return cirq.unitary(cirq.CNOT)

        @property
        def qubits(self):
            return self.qs

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

    class NoDetails(cirq.Operation):

        def __init__(self, *qs: cirq.Qid):
            self.qs = qs

        @property
        def qubits(self):
            return self.qs

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()
    a, b, c = cirq.LineQubit.range(3)
    assert not cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(b, a))
    assert not cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(b, c))
    assert cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(c, b))
    assert cirq.commutes(CustomCnotOp(a, b), CustomCnotOp(a, b))
    assert cirq.commutes(CustomCnotOp(a, b), NoDetails(c))
    assert cirq.commutes(CustomCnotOp(a, b), NoDetails(a), default=None) is None
    assert cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(a, b))
    assert cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(a, b))
    assert cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(c, b))
    assert cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(c, b))
    assert not cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(b, a))
    assert not cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(b, a))
    assert not cirq.commutes(custom_cnot_gate(a, b), CustomCnotOp(b, c))
    assert not cirq.commutes(custom_cnot_gate(a, b), custom_cnot_gate(b, c))