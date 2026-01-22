import abc
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
from cirq import protocols
from cirq._doc import document
def projector(self, qubit_order: Optional['cirq.QubitOrder']=None) -> np.ndarray:
    """The projector associated with this state expressed as a matrix.

        This is |s⟩⟨s| where |s⟩ is this state.
        """
    from cirq import ops
    if qubit_order is None:
        qubit_order = ops.QubitOrder.DEFAULT
    qubit_order = ops.QubitOrder.as_qubit_order(qubit_order)
    qubits = qubit_order.order_for(self.qubits)
    mat = np.ones(1, dtype=np.complex128)
    for qubit in qubits:
        oneq_state = self[qubit]
        oneq_proj = oneq_state.projector()
        mat = np.kron(mat, oneq_proj)
    return mat