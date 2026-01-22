from typing import Optional
import numpy as np
import pytest
import cirq
from cirq import testing
def test_unitary_from_apply_unitary():

    class ApplyGate(cirq.Gate):

        def num_qubits(self):
            return 1

        def _apply_unitary_(self, args):
            return cirq.apply_unitary(cirq.X(cirq.LineQubit(0)), args)

    class UnknownType:

        def _apply_unitary_(self, args):
            assert False

    class ApplyGateNotUnitary(cirq.Gate):

        def num_qubits(self):
            return 1

        def _apply_unitary_(self, args):
            return None

    class ApplyOp(cirq.Operation):

        def __init__(self, q):
            self.q = q

        @property
        def qubits(self):
            return (self.q,)

        def with_qubits(self, *new_qubits):
            return ApplyOp(*new_qubits)

        def _apply_unitary_(self, args):
            return cirq.apply_unitary(cirq.X(self.q), args)
    assert cirq.has_unitary(ApplyGate())
    assert cirq.has_unitary(ApplyOp(cirq.LineQubit(0)))
    assert not cirq.has_unitary(ApplyGateNotUnitary())
    assert not cirq.has_unitary(UnknownType())
    np.testing.assert_allclose(cirq.unitary(ApplyGate()), np.array([[0, 1], [1, 0]]))
    np.testing.assert_allclose(cirq.unitary(ApplyOp(cirq.LineQubit(0))), np.array([[0, 1], [1, 0]]))
    assert cirq.unitary(ApplyGateNotUnitary(), default=None) is None
    assert cirq.unitary(UnknownType(), default=None) is None