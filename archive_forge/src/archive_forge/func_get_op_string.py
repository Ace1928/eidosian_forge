from typing import List, Dict, Sequence, Any
import cirq
import cirq_pasqal
def get_op_string(self, cirq_op: cirq.Operation) -> str:
    """Find the string representation for a given operation.

        Args:
            cirq_op: A cirq operation.

        Returns:
            String representing the gate operations.

        Raises:
            ValueError: If the operations gate is not supported.
        """
    if not self.device.is_pasqal_device_op(cirq_op) or isinstance(cirq_op.gate, cirq.MeasurementGate):
        raise ValueError('Got unknown operation:', cirq_op)
    return str(cirq_op.gate)