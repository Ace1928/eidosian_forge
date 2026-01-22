import pytest
import numpy as np
import cirq
from cirq.testing.circuit_compare import _assert_apply_unitary_works_when_axes_transposed
def test_assert_has_consistent_qid_shape():

    class ConsistentGate(cirq.Gate):

        def _num_qubits_(self):
            return 4

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class InconsistentGate(cirq.Gate):

        def _num_qubits_(self):
            return 2

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class BadShapeGate(cirq.Gate):

        def _num_qubits_(self):
            return 4

        def _qid_shape_(self):
            return (1, 2, 0, 4)

    class ConsistentOp(cirq.Operation):

        def with_qubits(self, *qubits):
            raise NotImplementedError

        @property
        def qubits(self):
            return cirq.LineQubit.range(4)

        def _num_qubits_(self):
            return 4

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class InconsistentOp1(cirq.Operation):

        def with_qubits(self, *qubits):
            raise NotImplementedError

        @property
        def qubits(self):
            return cirq.LineQubit.range(2)

        def _num_qubits_(self):
            return 4

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class InconsistentOp2(cirq.Operation):

        def with_qubits(self, *qubits):
            raise NotImplementedError

        @property
        def qubits(self):
            return cirq.LineQubit.range(4)

        def _num_qubits_(self):
            return 2

        def _qid_shape_(self):
            return (1, 2, 3, 4)

    class InconsistentOp3(cirq.Operation):

        def with_qubits(self, *qubits):
            raise NotImplementedError

        @property
        def qubits(self):
            return cirq.LineQubit.range(4)

        def _num_qubits_(self):
            return 4

        def _qid_shape_(self):
            return (1, 2)

    class NoProtocol:
        pass
    cirq.testing.assert_has_consistent_qid_shape(ConsistentGate())
    with pytest.raises(AssertionError, match='disagree'):
        cirq.testing.assert_has_consistent_qid_shape(InconsistentGate())
    with pytest.raises(AssertionError, match='positive'):
        cirq.testing.assert_has_consistent_qid_shape(BadShapeGate())
    cirq.testing.assert_has_consistent_qid_shape(ConsistentOp())
    with pytest.raises(AssertionError, match='disagree'):
        cirq.testing.assert_has_consistent_qid_shape(InconsistentOp1())
    with pytest.raises(AssertionError, match='disagree'):
        cirq.testing.assert_has_consistent_qid_shape(InconsistentOp2())
    with pytest.raises(AssertionError, match='disagree'):
        cirq.testing.assert_has_consistent_qid_shape(InconsistentOp3())
    cirq.testing.assert_has_consistent_qid_shape(NoProtocol())