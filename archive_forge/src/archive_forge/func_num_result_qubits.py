from typing import Optional
from qiskit.circuit import QuantumCircuit
@property
def num_result_qubits(self) -> int:
    """The number of result qubits to limit the output to.

        Returns:
            The number of result qubits.
        """
    return self._num_result_qubits