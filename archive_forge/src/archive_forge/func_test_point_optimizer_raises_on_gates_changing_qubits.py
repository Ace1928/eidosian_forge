from typing import Optional, TYPE_CHECKING, Set, List
import pytest
import cirq
from cirq import PointOptimizer, PointOptimizationSummary, Operation
from cirq.testing import EqualsTester
def test_point_optimizer_raises_on_gates_changing_qubits():

    class EverythingIs42(cirq.PointOptimizer):
        """Changes all single qubit operations to act on LineQubit(42)"""

        def optimization_at(self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation') -> Optional['cirq.PointOptimizationSummary']:
            new_op = op
            if len(op.qubits) == 1 and isinstance(op, cirq.GateOperation):
                new_op = op.gate(cirq.LineQubit(42))
            return cirq.PointOptimizationSummary(clear_span=1, clear_qubits=op.qubits, new_operations=new_op)
    c = cirq.Circuit(cirq.X(cirq.LineQubit(0)), cirq.X(cirq.LineQubit(1)))
    with pytest.raises(ValueError, match='new qubits'):
        EverythingIs42().optimize_circuit(c)