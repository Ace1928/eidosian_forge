from typing import Any, Dict, Optional, Tuple, TYPE_CHECKING, Sequence
import numpy as np
import sympy
from cirq import protocols, value
from cirq._doc import document
from cirq.ops import raw_types
def identity_each(*qubits: 'cirq.Qid') -> 'cirq.Operation':
    """Returns a single IdentityGate applied to all the given qubits.

    Args:
        *qubits: The qubits that the identity gate will apply to.

    Returns:
        An identity operation on the given qubits.

    Raises:
        ValueError: If the qubits are not instances of Qid.
    """
    for qubit in qubits:
        if not isinstance(qubit, raw_types.Qid):
            raise ValueError(f'Not a cirq.Qid: {qubit!r}.')
    return IdentityGate(qid_shape=protocols.qid_shape(qubits)).on(*qubits)